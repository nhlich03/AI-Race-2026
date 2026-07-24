"""Preprocess a COLMAP scene so the vanilla 3DGS reader accepts it.

The VAR-2026 data has two things the stock gaussian-splatting COLMAP loader
cannot handle:

1. Camera model is SIMPLE_RADIAL (has a distortion coeff k). The 3DGS reader
   only accepts PINHOLE / SIMPLE_PINHOLE. k here is ~0.008 (negligible) and the
   test poses are given as pure pinhole, so we drop k and rewrite as PINHOLE.
2. images.bin registers MORE images than exist on disk (it includes the held-out
   test views, whose image files are not shipped). The reader would crash trying
   to open a missing file, so we keep only images whose file is present.

Output: a writable source dir with images/ (symlinked), sparse/0/cameras.bin +
images.bin rewritten, points3D.bin reused as-is. Point train.py at this dir.
"""
import os
import struct
import shutil

# COLMAP camera model -> number of params
_NPARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}


def _read_cameras_bin(path):
    cams = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cam_id, model_id = struct.unpack("<ii", f.read(8))
            w, h = struct.unpack("<QQ", f.read(16))
            k = _NPARAMS[model_id]
            params = struct.unpack("<" + "d" * k, f.read(8 * k))
            cams[cam_id] = (model_id, w, h, params)
    return cams


def _model_to_pinhole_params(model_id, params):
    """Return (fx, fy, cx, cy) dropping any distortion."""
    if model_id == 0:      # SIMPLE_PINHOLE: f, cx, cy
        f, cx, cy = params[:3]
        return f, f, cx, cy
    if model_id == 1:      # PINHOLE: fx, fy, cx, cy
        return params[0], params[1], params[2], params[3]
    if model_id in (2, 3, 8, 9):  # SIMPLE_RADIAL / RADIAL / *_FISHEYE: f, cx, cy, ...
        f, cx, cy = params[0], params[1], params[2]
        return f, f, cx, cy
    if model_id in (4, 5, 6, 10):  # OPENCV / FULL_OPENCV / ...: fx, fy, cx, cy, ...
        return params[0], params[1], params[2], params[3]
    raise ValueError(f"Unsupported COLMAP model id {model_id}")


def _write_cameras_bin_pinhole(path, cams):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for cam_id, (model_id, w, h, params) in cams.items():
            fx, fy, cx, cy = _model_to_pinhole_params(model_id, params)
            f.write(struct.pack("<ii", cam_id, 1))          # model_id 1 = PINHOLE
            f.write(struct.pack("<QQ", w, h))
            f.write(struct.pack("<dddd", fx, fy, cx, cy))


def _read_images_bin(path):
    """Yield raw image records so we can rewrite them verbatim (minus filtering)."""
    records = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            image_id = struct.unpack("<i", f.read(4))[0]
            qvec = f.read(8 * 4)
            tvec = f.read(8 * 3)
            cam_id = struct.unpack("<i", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00" or c == b"":
                    break
                name += c
            npts = struct.unpack("<Q", f.read(8))[0]
            pts = f.read(npts * (8 + 8 + 8))  # x(d), y(d), point3D_id(int64)
            records.append(dict(image_id=image_id, qvec=qvec, tvec=tvec,
                                cam_id=cam_id, name=name, npts=npts, pts=pts))
    return records


def _write_images_bin(path, records):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(records)))
        for r in records:
            f.write(struct.pack("<i", r["image_id"]))
            f.write(r["qvec"])
            f.write(r["tvec"])
            f.write(struct.pack("<i", r["cam_id"]))
            f.write(r["name"] + b"\x00")
            f.write(struct.pack("<Q", r["npts"]))
            f.write(r["pts"])


def _link_or_copy(src, dst):
    if os.path.lexists(dst):
        return
    try:
        os.symlink(src, dst)
    except OSError:
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def prepare_source(src_train_dir, out_dir):
    """src_train_dir has images/ + sparse/0/. Returns a prepared source dir."""
    src_images = os.path.join(src_train_dir, "images")
    src_sparse = os.path.join(src_train_dir, "sparse", "0")

    present = {f for f in os.listdir(src_images)}
    out_sparse = os.path.join(out_dir, "sparse", "0")
    os.makedirs(out_sparse, exist_ok=True)

    # images/ -> symlink (avoid copying hundreds of MB)
    _link_or_copy(os.path.abspath(src_images), os.path.join(out_dir, "images"))

    # cameras.bin -> PINHOLE
    cams = _read_cameras_bin(os.path.join(src_sparse, "cameras.bin"))
    _write_cameras_bin_pinhole(os.path.join(out_sparse, "cameras.bin"), cams)

    # images.bin -> keep only records whose image file exists on disk
    records = _read_images_bin(os.path.join(src_sparse, "images.bin"))
    kept = [r for r in records if os.path.basename(r["name"].decode("utf-8", "replace")) in present]
    _write_images_bin(os.path.join(out_sparse, "images.bin"), kept)

    # points3D.bin -> reuse as-is
    _link_or_copy(os.path.abspath(os.path.join(src_sparse, "points3D.bin")),
                  os.path.join(out_sparse, "points3D.bin"))

    dropped = len(records) - len(kept)
    print(f"[prepare] {src_train_dir}: cameras->PINHOLE, "
          f"images {len(records)}->{len(kept)} (dropped {dropped} missing-file views)")
    return out_dir


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="scene train/ dir (images/ + sparse/0/)")
    ap.add_argument("--out", required=True, help="output prepared source dir")
    a = ap.parse_args()
    prepare_source(a.src, a.out)

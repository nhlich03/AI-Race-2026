"""Low-level COLMAP binary IO shared by prepare.py and the experiment harness.

Handles cameras.bin, images.bin, points3D.bin (classic COLMAP format) plus a few
helpers: converting any camera model to PINHOLE intrinsics, unpacking a pose from
an images.bin record, and reading/writing/augmenting the sparse point cloud.
"""
import os
import struct
import shutil

# COLMAP camera model id -> number of intrinsic params
_NPARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}


# ----------------------------------------------------------------------------- cameras
def read_cameras_bin(path):
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


def model_to_pinhole_params(model_id, params):
    """Return (fx, fy, cx, cy) for any model, dropping distortion."""
    if model_id == 0:      # SIMPLE_PINHOLE: f, cx, cy
        f, cx, cy = params[:3]
        return f, f, cx, cy
    if model_id == 1:      # PINHOLE: fx, fy, cx, cy
        return params[0], params[1], params[2], params[3]
    if model_id in (2, 3, 8, 9):   # SIMPLE_RADIAL / RADIAL / *_FISHEYE: f, cx, cy, ...
        f, cx, cy = params[0], params[1], params[2]
        return f, f, cx, cy
    if model_id in (4, 5, 6, 10):  # OPENCV / FULL_OPENCV / ...: fx, fy, cx, cy, ...
        return params[0], params[1], params[2], params[3]
    raise ValueError(f"Unsupported COLMAP model id {model_id}")


def write_cameras_bin_pinhole(path, cams):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for cam_id, (model_id, w, h, params) in cams.items():
            fx, fy, cx, cy = model_to_pinhole_params(model_id, params)
            f.write(struct.pack("<ii", cam_id, 1))      # 1 = PINHOLE
            f.write(struct.pack("<QQ", w, h))
            f.write(struct.pack("<dddd", fx, fy, cx, cy))


def camera_intrinsics(cams, cam_id):
    """(w, h, fx, fy, cx, cy) for a camera id (PINHOLE-ized)."""
    model_id, w, h, params = cams[cam_id]
    fx, fy, cx, cy = model_to_pinhole_params(model_id, params)
    return w, h, fx, fy, cx, cy


# ----------------------------------------------------------------------------- images
def read_images_bin(path):
    """Return raw image records so we can rewrite them verbatim (minus filtering)."""
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


def write_images_bin(path, records):
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


def record_name(record):
    return os.path.basename(record["name"].decode("utf-8", "replace"))


def unpack_pose(record):
    """(qw, qx, qy, qz, tx, ty, tz) from a raw images.bin record (COLMAP world-to-camera)."""
    qw, qx, qy, qz = struct.unpack("<dddd", record["qvec"])
    tx, ty, tz = struct.unpack("<ddd", record["tvec"])
    return qw, qx, qy, qz, tx, ty, tz


# ----------------------------------------------------------------------------- points3D
def read_points3D_bin(path):
    """Return list of point dicts; track kept as raw bytes so we can rewrite verbatim."""
    pts = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            pid = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<ddd", f.read(24))
            rgb = struct.unpack("<BBB", f.read(3))
            err = struct.unpack("<d", f.read(8))[0]
            tlen = struct.unpack("<Q", f.read(8))[0]
            track = f.read(tlen * 8)  # (image_id int32, point2D_idx int32) * tlen
            pts.append(dict(id=pid, xyz=xyz, rgb=rgb, err=err, tlen=tlen, track=track))
    return pts


def write_points3D_bin(path, pts):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(pts)))
        for p in pts:
            f.write(struct.pack("<Q", p["id"]))
            f.write(struct.pack("<ddd", *p["xyz"]))
            f.write(struct.pack("<BBB", *p["rgb"]))
            f.write(struct.pack("<d", p["err"]))
            f.write(struct.pack("<Q", p["tlen"]))
            f.write(p["track"])


def augment_random_points(pts, n_extra, rng):
    """Append n_extra random points inside the existing bbox with random RGB.

    rng is a random.Random instance (deterministic). Track is empty (length 0),
    which is fine — 3DGS only uses xyz+rgb to seed the initial Gaussians.
    """
    if n_extra <= 0 or not pts:
        return pts
    xs = [p["xyz"][0] for p in pts]
    ys = [p["xyz"][1] for p in pts]
    zs = [p["xyz"][2] for p in pts]
    lo = (min(xs), min(ys), min(zs))
    hi = (max(xs), max(ys), max(zs))
    next_id = max(p["id"] for p in pts) + 1
    for i in range(n_extra):
        xyz = tuple(lo[d] + rng.random() * (hi[d] - lo[d]) for d in range(3))
        rgb = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        pts.append(dict(id=next_id + i, xyz=xyz, rgb=rgb, err=1.0, tlen=0, track=b""))
    return pts


# ----------------------------------------------------------------------------- fs util
def link_or_copy(src, dst):
    if os.path.lexists(dst):
        return
    try:
        os.symlink(src, dst)
    except OSError:
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

"""Prepare a COLMAP scene for the vanilla 3DGS reader, with optional holdout + augment.

Base behaviour (holdout_frac=0, augment=None) is identical to the original
prepare_pinhole: convert cameras to PINHOLE (drop tiny distortion) and drop
images.bin records whose image file is not shipped.

Extra modes for the experiment harness:
- holdout_frac > 0: withhold every Nth view (N≈1/frac) from training and emit their
  poses as a test_poses.csv-style file, so we can render them and compare to the
  real (withheld) image → measurable PSNR/SSIM/LPIPS on a scene that has no GT test set.
- augment={"random_points": N}: inject N random points into the init point cloud
  (helps sparse regions like thin antennas — an experiment to A/B).
"""
import os
import csv
import random

import colmap_io as cio


def _write_holdout_csv(path, heldout, cams):
    """heldout: list of images.bin records. Writes test_poses.csv column format."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_name", "qw", "qx", "qy", "qz", "tx", "ty", "tz",
                    "fx", "fy", "cx", "cy", "width", "height"])
        for r in heldout:
            qw, qx, qy, qz, tx, ty, tz = cio.unpack_pose(r)
            width, height, fx, fy, cx, cy = cio.camera_intrinsics(cams, r["cam_id"])
            w.writerow([cio.record_name(r), qw, qx, qy, qz, tx, ty, tz,
                        fx, fy, cx, cy, width, height])


def prepare_source(src_train_dir, out_dir, holdout_frac=0.0, seed=0, augment=None):
    """src_train_dir has images/ + sparse/0/. Returns a dict:
        {"source": out_dir, "holdout_csv": path_or_None, "holdout_gt": {name: gt_path}}
    """
    src_images = os.path.join(src_train_dir, "images")
    src_sparse = os.path.join(src_train_dir, "sparse", "0")
    out_sparse = os.path.join(out_dir, "sparse", "0")
    os.makedirs(out_sparse, exist_ok=True)

    present = set(os.listdir(src_images))
    cio.link_or_copy(os.path.abspath(src_images), os.path.join(out_dir, "images"))

    # cameras.bin -> PINHOLE
    cams = cio.read_cameras_bin(os.path.join(src_sparse, "cameras.bin"))
    cio.write_cameras_bin_pinhole(os.path.join(out_sparse, "cameras.bin"), cams)

    # images.bin -> keep only records whose file exists, sorted by name for determinism
    records = cio.read_images_bin(os.path.join(src_sparse, "images.bin"))
    kept = [r for r in records if cio.record_name(r) in present]
    kept.sort(key=cio.record_name)

    holdout_csv, holdout_gt = None, {}
    if holdout_frac and holdout_frac > 0:
        n = max(2, round(1.0 / holdout_frac))               # frac 0.125 -> every 8th
        heldout = [r for i, r in enumerate(kept) if i % n == 0]
        train = [r for i, r in enumerate(kept) if i % n != 0]
        cio.write_images_bin(os.path.join(out_sparse, "images.bin"), train)
        holdout_csv = os.path.join(out_dir, "holdout_poses.csv")
        _write_holdout_csv(holdout_csv, heldout, cams)
        holdout_gt = {cio.record_name(r): os.path.join(src_images, cio.record_name(r))
                      for r in heldout}
        print(f"[prepare] {src_train_dir}: cameras->PINHOLE, train={len(train)} "
              f"holdout={len(heldout)} (dropped {len(records) - len(kept)} missing-file)")
    else:
        cio.write_images_bin(os.path.join(out_sparse, "images.bin"), kept)
        print(f"[prepare] {src_train_dir}: cameras->PINHOLE, "
              f"images {len(records)}->{len(kept)} (dropped {len(records) - len(kept)} missing-file)")

    # points3D.bin -> reuse as-is, or augment with random points
    n_extra = (augment or {}).get("random_points", 0)
    dst_pts = os.path.join(out_sparse, "points3D.bin")
    if n_extra > 0:
        pts = cio.read_points3D_bin(os.path.join(src_sparse, "points3D.bin"))
        before = len(pts)
        cio.augment_random_points(pts, n_extra, random.Random(seed))
        cio.write_points3D_bin(dst_pts, pts)
        print(f"[prepare] points3D {before} -> {len(pts)} (+{n_extra} random)")
    else:
        cio.link_or_copy(os.path.abspath(os.path.join(src_sparse, "points3D.bin")), dst_pts)

    return {"source": out_dir, "holdout_csv": holdout_csv, "holdout_gt": holdout_gt}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout_frac", type=float, default=0.0)
    ap.add_argument("--random_points", type=int, default=0)
    a = ap.parse_args()
    aug = {"random_points": a.random_points} if a.random_points else None
    r = prepare_source(a.src, a.out, holdout_frac=a.holdout_frac, augment=aug)
    print(r["source"], "| holdout_csv:", r["holdout_csv"], "| n_holdout:", len(r["holdout_gt"]))

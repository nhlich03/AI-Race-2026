"""Render a trained 3D Gaussian Splatting model at the exact camera poses
listed in a VAR-2026 test_poses.csv file.

The official gaussian-splatting repo can only render cameras that came from its
COLMAP loader. Here we build cameras directly from the CSV (COLMAP world-to-camera
convention) and drive the rasterizer through the repo's `MiniCam` helper.

CSV columns (COLMAP convention):
    image_name, qw, qx, qy, qz, tx, ty, tz, fx, fy, cx, cy, width, height

NOTE (baseline limitation): the stock diff-gaussian-rasterizer assumes the
principal point is at the image center. cx/cy from the CSV are therefore ignored
here. If they are far from (width/2, height/2) this introduces a small offset;
see README "Hướng cải tiến" for how to fix it.
"""
import os
import sys
import csv
import argparse
import numpy as np
import torch
from PIL import Image


def qvec2rotmat(qw, qx, qy, qz):
    """COLMAP quaternion -> world-to-camera rotation matrix (same as COLMAP/3DGS)."""
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def focal2fov(focal, pixels):
    return 2.0 * np.arctan(pixels / (2.0 * focal))


def build_reader(fp):
    """DictReader tolerant of spaces in the header ('image_name, qw, ...')."""
    reader = csv.DictReader(fp, skipinitialspace=True)
    for raw in reader:
        yield {(k.strip() if k else k): (v.strip() if isinstance(v, str) else v)
               for k, v in raw.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gs_repo", required=True, help="Path to the cloned gaussian-splatting repo")
    ap.add_argument("--model", required=True, help="Trained model dir (contains point_cloud/iteration_*/point_cloud.ply)")
    ap.add_argument("--poses", required=True, help="test_poses.csv")
    ap.add_argument("--out", required=True, help="Output dir for rendered PNGs")
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--white_bg", action="store_true", help="Use white background instead of black")
    ap.add_argument("--name_mode", choices=["exact", "png"], default="exact",
                    help="'exact' = write the CSV image_name verbatim (content is still PNG); "
                         "'png' = replace the extension with .png")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.gs_repo))
    from scene.cameras import MiniCam
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix

    os.makedirs(args.out, exist_ok=True)

    ply = os.path.join(args.model, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
    if not os.path.isfile(ply):
        # fall back to whatever iteration was actually saved
        pc_root = os.path.join(args.model, "point_cloud")
        iters = sorted(int(d.split("_")[1]) for d in os.listdir(pc_root) if d.startswith("iteration_"))
        if not iters:
            raise FileNotFoundError(f"No point_cloud/iteration_* found under {args.model}")
        ply = os.path.join(pc_root, f"iteration_{iters[-1]}", "point_cloud.ply")
        print(f"[render] iteration {args.iteration} not found, using {ply}")

    gaussians = GaussianModel(args.sh_degree)
    gaussians.load_ply(ply)

    class Pipe:
        convert_SHs_python = False
        compute_cov3D_python = False
        debug = False
        antialiasing = False  # present in newer repo versions; harmless otherwise
    pipe = Pipe()

    bg = torch.tensor([1.0, 1.0, 1.0] if args.white_bg else [0.0, 0.0, 0.0],
                      dtype=torch.float32, device="cuda")

    znear, zfar = 0.01, 100.0
    n = 0
    with open(args.poses, newline="") as fp:
        for row in build_reader(fp):
            name = row["image_name"]
            qw, qx, qy, qz = (float(row[k]) for k in ("qw", "qx", "qy", "qz"))
            tx, ty, tz = (float(row[k]) for k in ("tx", "ty", "tz"))
            fx, fy = float(row["fx"]), float(row["fy"])
            W, H = int(float(row["width"])), int(float(row["height"]))

            Rcw = qvec2rotmat(qw, qx, qy, qz)
            R = Rcw.T                       # 3DGS Camera expects the transposed (c2w) rotation
            T = np.array([tx, ty, tz], dtype=np.float64)
            FoVx = focal2fov(fx, W)
            FoVy = focal2fov(fy, H)

            w2c = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
            proj = getProjectionMatrix(znear, zfar, FoVx, FoVy).transpose(0, 1).cuda()
            full = (w2c.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)

            cam = MiniCam(W, H, FoVy, FoVx, znear, zfar, w2c, full)
            with torch.no_grad():
                img = render(cam, gaussians, pipe, bg)["render"]
            arr = (torch.clamp(img, 0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)

            fname = name if args.name_mode == "exact" else os.path.splitext(name)[0] + ".png"
            # Always encode PNG (lossless) regardless of the filename extension.
            Image.fromarray(arr).save(os.path.join(args.out, fname), format="PNG")
            n += 1

    print(f"[render] wrote {n} images to {args.out}")


if __name__ == "__main__":
    main()

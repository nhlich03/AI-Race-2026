"""Train 3DGS on every scene and render the requested test poses.

Expected data layout (each scene has its own train/ + test/):

    <data_root>/
    ├── scene_001/
    │   ├── train/
    │   │   ├── images/
    │   │   └── sparse/0/{cameras,images,points3D}.bin
    │   └── test/test_poses.csv
    ├── scene_002/
    │   └── ...

Produces:
    <output>/scene_xxx/           trained models
    <submission>/scene_xxx/*.png  images to be zipped for submission
"""
import os
import sys
import glob
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_pinhole import prepare_source


def find_scenes(data_root):
    scenes = []
    for entry in sorted(os.listdir(data_root)):
        scene_dir = os.path.join(data_root, entry)
        if os.path.isdir(os.path.join(scene_dir, "train")):
            scenes.append((entry, scene_dir))
    return scenes


def find_poses_csv(scene_dir):
    for cand in (
        os.path.join(scene_dir, "test", "test_poses.csv"),
        os.path.join(scene_dir, "test_poses.csv"),
    ):
        if os.path.isfile(cand):
            return cand
    hits = glob.glob(os.path.join(scene_dir, "**", "test_pose*.csv"), recursive=True)
    return hits[0] if hits else None


def run(cmd, env=None):
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gs_repo", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--output", default="output")
    ap.add_argument("--submission", default="submission")
    ap.add_argument("--prepared", default="prepared",
                    help="Where to write PINHOLE-converted, file-filtered source dirs")
    ap.add_argument("--iterations", type=int, default=30000,
                    help="Use 7000 for a fast first baseline, 30000 for the full run")
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--data_device", default="cpu", help="Keep training images on CPU to save VRAM")
    ap.add_argument("--white_bg", action="store_true")
    ap.add_argument("--name_mode", choices=["exact", "png"], default="exact",
                    help="Output filename mode passed to render_test_poses.py")
    ap.add_argument("--img_format", choices=["png", "jpeg"], default="jpeg",
                    help="Output image format (jpeg keeps the zip under submission size limits)")
    ap.add_argument("--jpeg_quality", type=int, default=95)
    ap.add_argument("--eval", action="store_true",
                    help="Hold out 1/8 of train views for validation → logs PSNR/SSIM to "
                         "TensorBoard at milestones. Costs a bit of training data; OMIT for "
                         "the final full-data submission run.")
    ap.add_argument("--only", nargs="*", default=None, help="Restrict to these scene names")
    ap.add_argument("--skip_trained", action="store_true", help="Skip scenes whose model already exists")
    ap.add_argument("--gpu", default=None, help="Pin CUDA_VISIBLE_DEVICES (e.g. 0 or 1) for manual 2-GPU split")
    args = ap.parse_args()

    env = dict(os.environ)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    scenes = find_scenes(args.data_root)
    if args.only:
        scenes = [(n, d) for (n, d) in scenes if n in set(args.only)]
    if not scenes:
        print(f"No scenes found under {args.data_root}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(scenes)} scene(s): {[n for n, _ in scenes]}")

    here = os.path.dirname(os.path.abspath(__file__))
    render_script = os.path.join(here, "render_test_poses.py")

    for name, scene_dir in scenes:
        source = os.path.join(scene_dir, "train")
        model_out = os.path.join(args.output, name)
        sub_out = os.path.join(args.submission, name)
        poses = find_poses_csv(scene_dir)
        if poses is None:
            print(f"[warn] no test_poses.csv for {name}, skipping", file=sys.stderr)
            continue

        print(f"\n===== {name} =====")
        done_marker = os.path.join(model_out, "point_cloud", f"iteration_{args.iterations}", "point_cloud.ply")
        if args.skip_trained and os.path.isfile(done_marker):
            print(f"[train] {name} already trained, skipping")
        else:
            # Preprocess: SIMPLE_RADIAL -> PINHOLE + drop images.bin entries with no file.
            prepared_source = prepare_source(source, os.path.join(args.prepared, name))
            # train.py always logs the training loss curve to TensorBoard (model dir).
            train_cmd = [
                sys.executable, os.path.join(args.gs_repo, "train.py"),
                "-s", prepared_source,
                "-m", model_out,
                "--iterations", str(args.iterations),
                "--sh_degree", str(args.sh_degree),
                "--data_device", args.data_device,
                "--save_iterations", str(args.iterations),
            ]
            if args.eval:
                # Hold out 1/8 of views; evaluate at milestones so TensorBoard also gets
                # validation PSNR/SSIM curves (not just training loss).
                it = args.iterations
                milestones = sorted({it // 4, it // 2, (it * 3) // 4, it} - {0})
                train_cmd += ["--eval", "--test_iterations", *map(str, milestones)]
            else:
                # No holdout: all images train (best for final submission). No val curve.
                train_cmd += ["--test_iterations", "-1"]
            run(train_cmd, env=env)

        render_cmd = [
            sys.executable, render_script,
            "--gs_repo", args.gs_repo,
            "--model", model_out,
            "--poses", poses,
            "--out", sub_out,
            "--iteration", str(args.iterations),
            "--sh_degree", str(args.sh_degree),
            "--name_mode", args.name_mode,
            "--img_format", args.img_format,
            "--jpeg_quality", str(args.jpeg_quality),
        ]
        if args.white_bg:
            render_cmd.append("--white_bg")
        run(render_cmd, env=env)

    print("\nAll scenes done. Package with scripts/make_submission.py")


if __name__ == "__main__":
    main()

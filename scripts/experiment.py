"""Run a named experiment across scenes in one of two modes.

  holdout    : withhold 1/8 of each scene's views (they have GT), train on the rest,
               render the withheld poses, measure PSNR/SSIM/LPIPS + composite score,
               and append rows to results/metrics.csv. Use to COMPARE experiments.
  submission : train on all views, render test_poses.csv, package a submission zip.
               Use with the winning config to produce the final answer.

The 3DGS repo in third_party/ is never modified — configs only pass flags.

Examples:
  python scripts/experiment.py --exp baseline --mode holdout --gs_repo third_party/gaussian-splatting \
      --data_root ~/varace/data --only HCM0421
  python scripts/experiment.py --exp densify  --mode submission --gs_repo third_party/gaussian-splatting \
      --data_root ~/varace/data
"""
import os
import sys
import csv
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_all import find_scenes, find_poses_csv, run   # reuse scene discovery + subprocess helper
from prepare import prepare_source
import experiments as registry


def _append_result(results_csv, row):
    os.makedirs(os.path.dirname(results_csv) or ".", exist_ok=True)
    new = not os.path.isfile(results_csv)
    with open(results_csv, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["exp", "scene", "mode", "n", "psnr", "ssim", "lpips", "psnr_norm", "score"])
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="experiment name (see experiments.py)")
    ap.add_argument("--mode", choices=["holdout", "submission"], required=True)
    ap.add_argument("--gs_repo", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--iterations", type=int, default=None, help="override config iterations (e.g. smoke test)")
    ap.add_argument("--holdout_frac", type=float, default=0.125)
    ap.add_argument("--data_device", default="cuda")
    ap.add_argument("--img_format", choices=["png", "jpeg"], default="jpeg")
    ap.add_argument("--jpeg_quality", type=int, default=95)
    ap.add_argument("--psnr_max", type=float, default=40.0)
    ap.add_argument("--results", default="results/metrics.csv")
    ap.add_argument("--workroot", default=".", help="root for output/prepared/eval_pred/submission dirs")
    ap.add_argument("--sub_name", default=None,
                    help="Submission namespace (default = exp name). Use the SAME --sub_name across "
                         "several submission runs to merge per-scene-type configs into one zip.")
    args = ap.parse_args()

    cfg = registry.get(args.exp)
    iterations = args.iterations if args.iterations is not None else cfg["iterations"]
    sub_name = args.sub_name or args.exp   # submission namespace (merge several exps into one zip)

    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    scenes = find_scenes(args.data_root)
    if args.only:
        scenes = [(n, d) for (n, d) in scenes if n in set(args.only)]
    if not scenes:
        sys.exit(f"No scenes under {args.data_root}")
    print(f"[exp={args.exp} mode={args.mode} iters={iterations}] scenes: {[n for n, _ in scenes]}")

    here = os.path.dirname(os.path.abspath(__file__))
    render_script = os.path.join(here, "render_test_poses.py")
    W = args.workroot

    def d(*p):
        return os.path.join(W, *p)

    for name, scene_dir in scenes:
        source = os.path.join(scene_dir, "train")
        model_out = d("output", args.exp, name)
        prepared_dir = d("prepared", args.exp, name)
        print(f"\n===== {args.exp} / {name} =====")

        prep = prepare_source(
            source, prepared_dir,
            holdout_frac=(args.holdout_frac if args.mode == "holdout" else 0.0),
            seed=0, augment=(cfg["prepare"] or None),
        )

        train_cmd = [
            sys.executable, os.path.join(args.gs_repo, "train.py"),
            "-s", prep["source"], "-m", model_out,
            "--iterations", str(iterations), "--sh_degree", str(cfg["sh_degree"]),
            "--data_device", args.data_device,
            "--test_iterations", "-1", "--save_iterations", str(iterations),
        ] + cfg["train_args"]
        run(train_cmd, env=env)

        common_render = [
            sys.executable, render_script, "--gs_repo", args.gs_repo, "--model", model_out,
            "--iteration", str(iterations), "--sh_degree", str(cfg["sh_degree"]),
            "--name_mode", "exact", "--img_format", args.img_format,
            "--jpeg_quality", str(args.jpeg_quality),
        ]

        if args.mode == "holdout":
            pred_dir = d("eval_pred", args.exp, name)
            run(common_render + ["--poses", prep["holdout_csv"], "--out", pred_dir], env=env)
            import eval_metrics
            r = eval_metrics.score(pred_dir, prep["holdout_gt"], psnr_max=args.psnr_max)
            if r is None:
                print(f"[eval] {name}: no matched images")
            else:
                print(f"[eval] {name}: score={r['score']} psnr={r['psnr']:.2f} "
                      f"ssim={r['ssim']:.4f} lpips={r['lpips']} (n={r['n']})")
                _append_result(args.results, [args.exp, name, args.mode, r["n"],
                                              f"{r['psnr']:.4f}", f"{r['ssim']:.5f}",
                                              "" if r["lpips"] is None else f"{r['lpips']:.5f}",
                                              f"{r['psnr_norm']:.5f}",
                                              "" if r["score"] is None else f"{r['score']:.5f}"])
        else:  # submission
            poses = find_poses_csv(scene_dir)
            if poses is None:
                print(f"[warn] no test_poses.csv for {name}, skipping render", file=sys.stderr)
                continue
            sub_dir = d("submission", sub_name, name)
            run(common_render + ["--poses", poses, "--out", sub_dir], env=env)

    if args.mode == "submission":
        sub_root = d("submission", sub_name)
        out_zip = d(f"submission_{sub_name}.zip")
        run([sys.executable, os.path.join(here, "make_submission.py"),
             "--submission", sub_root, "--out", out_zip], env=env)
        print(f"\nSubmission zip: {out_zip}")
    else:
        print(f"\nHoldout results appended to {args.results}")


if __name__ == "__main__":
    main()

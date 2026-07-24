"""Self-evaluate on the PUBLIC set, where ground-truth test images are available.

Computes the three official metrics and the combined VAR-2026 score:

    Score = 0.4 * (1 - LPIPS) + 0.3 * SSIM + 0.3 * PSNR_norm
    PSNR_norm = clamp(PSNR / PSNR_max, 0, 1)

PSNR_max is the organizer's undisclosed normalization threshold; pass --psnr_max
to match whatever value you assume (default 40).

Usage:
    python scripts/evaluate.py --pred submission/scene_001 --gt <path/to/gt_images>
    # or evaluate every scene at once by pointing at the parent dirs:
    python scripts/evaluate.py --pred_root submission --gt_root <public_gt_root>
"""
import os
import argparse
import numpy as np
import torch
from PIL import Image

import lpips  # pip install lpips
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr


def load_rgb(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def match_gt(pred_name, gt_dir):
    stem = os.path.splitext(pred_name)[0]
    for ext in (".png", ".jpg", ".jpeg", ".JPG", ".PNG"):
        cand = os.path.join(gt_dir, stem + ext)
        if os.path.isfile(cand):
            return cand
    return None


def eval_scene(pred_dir, gt_dir, loss_fn, device, psnr_max):
    preds = sorted(f for f in os.listdir(pred_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    rows = []
    for name in preds:
        gt_path = match_gt(name, gt_dir)
        if gt_path is None:
            continue
        gt = load_rgb(gt_path)
        pr = load_rgb(os.path.join(pred_dir, name), size=(gt.shape[1], gt.shape[0]))

        psnr = sk_psnr(gt, pr, data_range=1.0)
        ssim = sk_ssim(gt, pr, channel_axis=2, data_range=1.0)
        with torch.no_grad():
            t_gt = torch.from_numpy(gt).permute(2, 0, 1)[None].to(device) * 2 - 1
            t_pr = torch.from_numpy(pr).permute(2, 0, 1)[None].to(device) * 2 - 1
            lp = loss_fn(t_gt, t_pr).item()
        rows.append((psnr, ssim, lp))

    if not rows:
        return None
    psnr, ssim, lp = np.mean(rows, axis=0)
    psnr_norm = float(np.clip(psnr / psnr_max, 0.0, 1.0))
    score = 0.4 * (1 - lp) + 0.3 * ssim + 0.3 * psnr_norm
    return dict(n=len(rows), psnr=psnr, ssim=ssim, lpips=lp, psnr_norm=psnr_norm, score=score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred")
    ap.add_argument("--gt")
    ap.add_argument("--pred_root")
    ap.add_argument("--gt_root")
    ap.add_argument("--psnr_max", type=float, default=40.0)
    ap.add_argument("--net", default="alex", choices=["alex", "vgg"], help="LPIPS backbone")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = lpips.LPIPS(net=args.net).to(device)

    def report(tag, r):
        if r is None:
            print(f"{tag:>16}: no matching GT images found")
        else:
            print(f"{tag:>16}: score={r['score']:.4f} | LPIPS={r['lpips']:.4f} "
                  f"SSIM={r['ssim']:.4f} PSNR={r['psnr']:.2f} (norm={r['psnr_norm']:.4f}) "
                  f"[{r['n']} imgs]")

    if args.pred and args.gt:
        report(os.path.basename(args.pred.rstrip("/\\")),
               eval_scene(args.pred, args.gt, loss_fn, device, args.psnr_max))
        return

    if args.pred_root and args.gt_root:
        scenes = sorted(d for d in os.listdir(args.pred_root)
                        if os.path.isdir(os.path.join(args.pred_root, d)))
        results = []
        for s in scenes:
            gt_dir = os.path.join(args.gt_root, s)
            # tolerate gt stored under <scene>/test/images or <scene>/images
            for sub in ("", "test/images", "images", "test"):
                cand = os.path.join(gt_dir, sub) if sub else gt_dir
                if os.path.isdir(cand) and any(f.lower().endswith((".png", ".jpg", ".jpeg"))
                                               for f in os.listdir(cand)):
                    gt_dir = cand
                    break
            r = eval_scene(os.path.join(args.pred_root, s), gt_dir, loss_fn, device, args.psnr_max)
            report(s, r)
            if r:
                results.append(r["score"])
        if results:
            print(f"{'MEAN':>16}: score={np.mean(results):.4f} over {len(results)} scenes")
        return

    ap.error("provide either --pred/--gt or --pred_root/--gt_root")


if __name__ == "__main__":
    main()

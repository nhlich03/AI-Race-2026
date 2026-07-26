"""Measure rendered holdout views against their ground-truth (withheld) images.

Computes the three official metrics and the VAR-2026 composite:
    score = 0.4*(1 - LPIPS) + 0.3*SSIM + 0.3*PSNR_norm,  PSNR_norm = clamp(PSNR/psnr_max, 0, 1)

PSNR_max is the organizer's undisclosed normalization threshold (default 40) — the
absolute score is only meaningful for RELATIVE comparison between experiments run
with the same psnr_max. LPIPS is optional (guarded import); without it, `lpips` and
`score` come back None and callers should rank on (ssim, psnr_norm) instead.
"""
import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr

try:
    import torch
    import lpips as _lpips_mod
    _HAS_LPIPS = True
except Exception:
    _HAS_LPIPS = False

_LPIPS_NET = None


def _get_lpips(net="alex"):
    global _LPIPS_NET
    if _LPIPS_NET is None:
        # Default to CPU: LPIPS conv layers hit cuDNN CUDNN_STATUS_NOT_INITIALIZED on
        # MIG GPUs, and eval is not the bottleneck (training dominates). Override with
        # LPIPS_DEVICE=cuda where cuDNN works fine.
        dev = os.environ.get("LPIPS_DEVICE", "cpu")
        _LPIPS_NET = _lpips_mod.LPIPS(net=net).to(dev).eval()
    return _LPIPS_NET


def _load_rgb(path, size=None):
    im = Image.open(path).convert("RGB")
    if size is not None and im.size != size:
        im = im.resize(size, Image.BICUBIC)
    return np.asarray(im, dtype=np.float32) / 255.0


def score(pred_dir, gt_map, psnr_max=40.0, net="alex"):
    """pred_dir holds rendered images named by image_name; gt_map: {image_name: gt_path}."""
    rows = []
    loss_fn = _get_lpips(net) if _HAS_LPIPS else None
    for name, gt_path in gt_map.items():
        pred_path = os.path.join(pred_dir, name)
        if not os.path.isfile(pred_path) or not os.path.isfile(gt_path):
            continue
        gt = _load_rgb(gt_path)
        pr = _load_rgb(pred_path, size=(gt.shape[1], gt.shape[0]))
        psnr = float(sk_psnr(gt, pr, data_range=1.0))
        ssim = float(sk_ssim(gt, pr, channel_axis=2, data_range=1.0))
        if loss_fn is not None:
            with torch.no_grad():
                dev = next(loss_fn.parameters()).device
                t_gt = torch.from_numpy(gt).permute(2, 0, 1)[None].to(dev) * 2 - 1
                t_pr = torch.from_numpy(pr).permute(2, 0, 1)[None].to(dev) * 2 - 1
                lp = float(loss_fn(t_gt, t_pr).item())
        else:
            lp = None
        rows.append((psnr, ssim, lp))

    if not rows:
        return None
    psnr = float(np.mean([r[0] for r in rows]))
    ssim = float(np.mean([r[1] for r in rows]))
    psnr_norm = float(np.clip(psnr / psnr_max, 0.0, 1.0))
    if _HAS_LPIPS:
        lpips_v = float(np.mean([r[2] for r in rows]))
        comp = 0.4 * (1 - lpips_v) + 0.3 * ssim + 0.3 * psnr_norm
    else:
        lpips_v, comp = None, None
    return dict(n=len(rows), psnr=psnr, ssim=ssim, lpips=lpips_v,
                psnr_norm=psnr_norm, score=comp)

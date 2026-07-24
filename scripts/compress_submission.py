"""Package a submission into a ZIP that fits under a size cap (default 350 MB).

The contest upload portal caps each submission at 350 MB and scenes cannot be
split across uploads (a missing scene is scored as 0). A lossless-PNG render of
all 386 target views is ~720 MB, so we re-encode to JPEG. This script:

  - reads images from a folder (scene_x/*.png|jpg) OR an existing .zip,
  - re-encodes each to JPEG at --quality (subsampling=0, optimize),
  - if still above --max_mb, steps quality down (95 -> 92 -> 90 -> ...) until it fits,
  - preserves the scene/filename layout exactly (keeps the CSV image_name).

JPEG q95 adds ~48 dB PSNR of error — far below the rendering error — so the
impact on the leaderboard metric is negligible.

Usage:
    python scripts/compress_submission.py --src submission --out submission_round1.zip
    python scripts/compress_submission.py --src old_png_submission.zip --out fixed.zip --max_mb 350
"""
import os
import io
import zipfile
import argparse
from PIL import Image

IMG_EXT = (".png", ".jpg", ".jpeg")


def iter_images_from_dir(root):
    for scene in sorted(os.listdir(root)):
        sdir = os.path.join(root, scene)
        if not os.path.isdir(sdir):
            continue
        for fn in sorted(os.listdir(sdir)):
            if fn.lower().endswith(IMG_EXT):
                with open(os.path.join(sdir, fn), "rb") as f:
                    yield f"{scene}/{fn}", f.read()


def iter_images_from_zip(path):
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if not n.endswith("/") and n.lower().endswith(IMG_EXT):
                yield n, z.read(n)


def build_zip(images, out, quality):
    """images: list of (arcname, raw_bytes). Returns size in MB."""
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for arcname, raw in images:
            im = Image.open(io.BytesIO(raw)).convert("RGB")
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=quality, optimize=True, subsampling=0)
            z.writestr(arcname, buf.getvalue())
    return os.path.getsize(out) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="submission folder OR an existing .zip")
    ap.add_argument("--out", default="submission_round1.zip")
    ap.add_argument("--max_mb", type=float, default=350.0, help="hard size cap (MB)")
    ap.add_argument("--quality", type=int, default=95, help="starting JPEG quality")
    ap.add_argument("--min_quality", type=int, default=80, help="lowest quality to try")
    args = ap.parse_args()

    if os.path.isdir(args.src):
        images = list(iter_images_from_dir(args.src))
    elif zipfile.is_zipfile(args.src):
        images = list(iter_images_from_zip(args.src))
    else:
        raise SystemExit(f"--src must be a folder or a .zip: {args.src}")

    if not images:
        raise SystemExit(f"No images found under {args.src}")

    scenes = sorted({a.split("/")[0] for a, _ in images})
    print(f"{len(images)} images across {len(scenes)} scenes: {scenes}")

    q = args.quality
    while q >= args.min_quality:
        mb = build_zip(images, args.out, q)
        status = "OK" if mb <= args.max_mb else "still too big"
        print(f"quality={q}: {mb:.1f} MB -> {status}")
        if mb <= args.max_mb:
            print(f"\nWrote {args.out} ({mb:.1f} MB, {len(images)} images, q{q})")
            return
        q -= 3

    print(f"\nWARNING: could not get under {args.max_mb} MB even at q{args.min_quality}. "
          f"Kept last attempt at {args.out}. Consider a lower --min_quality.")


if __name__ == "__main__":
    main()

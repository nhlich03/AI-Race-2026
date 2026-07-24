"""Zip a submission/ directory into submission_round1.zip with the layout:

    submission_round1.zip
    ├── scene_001/0001.png ...
    ├── scene_002/0001.png ...
"""
import os
import argparse
import zipfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default="submission", help="Dir containing scene_xxx/ subfolders")
    ap.add_argument("--out", default="submission_round1.zip")
    args = ap.parse_args()

    scenes = sorted(d for d in os.listdir(args.submission)
                    if os.path.isdir(os.path.join(args.submission, d)))
    if not scenes:
        raise SystemExit(f"No scene folders in {args.submission}")

    total = 0
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for scene in scenes:
            scene_dir = os.path.join(args.submission, scene)
            imgs = sorted(f for f in os.listdir(scene_dir)
                          if f.lower().endswith((".png", ".jpg", ".jpeg")))
            print(f"{scene}: {len(imgs)} images")
            for img in imgs:
                zf.write(os.path.join(scene_dir, img), arcname=f"{scene}/{img}")
                total += 1

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"\nWrote {args.out}  ({total} images, {size_mb:.1f} MB across {len(scenes)} scenes)")


if __name__ == "__main__":
    main()

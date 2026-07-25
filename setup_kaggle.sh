#!/usr/bin/env bash
# One-time environment setup on Kaggle (enable "Internet" + "GPU T4 x2" in the
# notebook settings first). Compiles the CUDA rasterizer against Kaggle's torch.
set -euo pipefail

# T4 = sm_75. If you switch to P100 use "6.0"; A4000 (contest ref HW) is "8.6".
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5}"

GS_DIR="third_party/gaussian-splatting"
# Pin a commit for reproducibility; override with GS_COMMIT=<hash> if the API drifts.
GS_COMMIT="${GS_COMMIT:-}"

if [ ! -d "$GS_DIR" ]; then
  git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting "$GS_DIR"
fi
if [ -n "$GS_COMMIT" ]; then
  git -C "$GS_DIR" checkout "$GS_COMMIT"
  git -C "$GS_DIR" submodule update --init --recursive
fi

pip install -q "$GS_DIR/submodules/diff-gaussian-rasterization"
pip install -q "$GS_DIR/submodules/simple-knn"
# fused-ssim is optional in newer repo versions; ignore if absent.
if [ -d "$GS_DIR/submodules/fused-ssim" ]; then
  pip install -q "$GS_DIR/submodules/fused-ssim" || echo "fused-ssim build skipped"
fi

pip install -q plyfile lpips scikit-image tensorboard

echo "Setup done. gaussian-splatting at: $GS_DIR"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())"

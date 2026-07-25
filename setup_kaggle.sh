#!/usr/bin/env bash
# One-time environment setup on Kaggle (enable "Internet" + "GPU T4 x2" in the
# notebook settings first). Compiles the CUDA rasterizer against Kaggle's torch.
set -euo pipefail

# Build the CUDA extensions for the GPU actually present.
# (T4=7.5, P100=6.0, A4000=8.6, A100=8.0, H100=9.0). Auto-detect from torch;
# override by exporting TORCH_CUDA_ARCH_LIST yourself before running.
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  export TORCH_CUDA_ARCH_LIST="$(python -c 'import torch;print("%d.%d"%torch.cuda.get_device_capability())' 2>/dev/null || echo 7.5)"
fi
echo "Building CUDA extensions for arch: $TORCH_CUDA_ARCH_LIST"

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

pip install -q plyfile lpips scikit-image

echo "Setup done. gaussian-splatting at: $GS_DIR"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())"

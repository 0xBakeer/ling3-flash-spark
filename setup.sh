#!/usr/bin/env bash
# setup.sh — build SGLang (inclusionAI ling_v3_support_mxfp4_humming) for GB10 / DGX Spark.
# Humming MoE + online FP8 LM head + DSPARK speculative decoding all live on this branch.
set -euo pipefail
RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$RECIPE_DIR"

BRANCH="${SGLANG_BRANCH:-ling_v3_support_mxfp4_humming}"
REPO="${SGLANG_REPO:-https://github.com/inclusionAI/sglang.git}"

echo "=== [1/4] uv ==="
command -v uv >/dev/null 2>&1 || python3 -m pip install --user -U uv --break-system-packages
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo "=== [2/4] venv (python 3.11) ==="
[[ -d .venv ]] || uv venv --python 3.11 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install --upgrade "openai>=1.52.0,<2.0.0"

echo "=== [3/4] clone $BRANCH ==="
if [[ -d sglang/.git ]]; then
    git -C sglang fetch origin "$BRANCH" --depth 1
    git -C sglang checkout -q FETCH_HEAD
else
    git clone --depth 1 -b "$BRANCH" "$REPO" sglang
fi
git -C sglang log --oneline -1

echo "=== [4/4] build sglang (no rust exts) ==="
export SGLANG_BUILD_RUST_EXTS=none
J=$(( $(nproc) < 8 ? $(nproc) : 8 ))
export MAX_JOBS="$J" NINJA_NUM_JOBS="$J" CMAKE_BUILD_PARALLEL_LEVEL="$J"
echo "MAX_JOBS=$J"
uv pip install -e "./sglang/python[all]"

echo "=== flashinfer cubin ==="
uv pip install flashinfer-cubin==0.6.17 --index-url https://flashinfer.ai/whl || \
  echo "WARN: flashinfer-cubin pin unavailable; relying on the sglang[all] resolution"

python -c "import torch, importlib.metadata as md; print('sglang', md.version('sglang')); print('torch', torch.__version__, torch.version.cuda); print('cap', torch.cuda.get_device_capability())"
echo "SETUP COMPLETE"

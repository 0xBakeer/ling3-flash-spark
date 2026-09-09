#!/usr/bin/env bash
# atlas/bench_ling.sh -- take inference-atlas speed rows for this recipe.
# Needs: a running ./start.sh (PROFILE=humming-dspark) and an inference-atlas
# checkout with atlas-bench (ATLAS_DIR). One packet + one run per workload.
# The atlas contract derives config_id/run_id from the args; never hand-edit.
set -u
export PATH=$HOME/.local/bin:$PATH
ATLAS_DIR="${ATLAS_DIR:-$HOME/inf-atlas}"
RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${BASE:-http://127.0.0.1:30000/v1}"
LOGIN="${ATLAS_LOGIN:-0xBakeer}"
SGL_VER="$(cd "$RECIPE_DIR" && source .venv/bin/activate && python -c 'import importlib.metadata as m;print(m.version("sglang"))')"
SGL_SHA="$(git -C "$RECIPE_DIR/sglang" rev-parse --short=7 HEAD)"
RECIPE_SHA="$(git -C "$RECIPE_DIR" rev-parse --short=7 HEAD 2>/dev/null || echo unversioned)"
CELL="sglang@${SGL_VER}/inclusionAI/Ling-3.0-flash/mxfp4/nvidia-gb10-dgx-spark"
BUILD="github.com/inclusionAI/sglang@${SGL_SHA}"
NOTES="Served with the ling3-flash-dgx-spark recipe (${RECIPE_SHA}), PROFILE=humming-dspark: inclusionAI/Ling-3.0-flash-fp4 (MXFP4 routed experts, FP8 attention/shared/router) with --moe-runner-backend humming, online FP8 LM head (SGLANG_ENABLE_FP8_LM_HEAD=1), DSpark speculative decoding with inclusionAI/Ling-3.0-flash-dspark and the KDA ReplaySSM exact fold (--enable-linear-replayssm-spec, ring 32), context 262144 native (no YaRN), mem-fraction 0.75, max-running 1, page 64, decode cuda graph bs 1, prefill cuda graph disabled, FlashInfer autotune off with precompiled CUTLASS MXFP4 kernels. Thinking off per request. Single request at a time on every row."
WORKLOADS=(serve-single-i256-o256-v1 prefill-8k-v1 prefill-32k-v1 prefill-128k-v1 longctx-needle-32k-v1 longctx-needle-128k-v1 longctx-depth-sweep-v1)
cd "$ATLAS_DIR/bench" || exit 1
mkdir -p logs packets; chmod 700 packets
for W in "${WORKLOADS[@]}"; do
  [ -e "logs/ling.$W.done" ] && { echo "skip $W"; continue; }
  curl -sf "$BASE/models" >/dev/null || { echo "!! server down before $W"; break; }
  echo "--- $W $(date -Is)" | tee -a logs/queue_ling.log
  uv run atlas-bench packet --cell "$CELL" --workload "$W" \
    --arg moe-runner-backend=humming --arg speculative-algorithm=DSPARK \
    --arg speculative-draft-model-path=inclusionAI/Ling-3.0-flash-dspark \
    --arg enable-linear-replayssm-spec=true --arg linear-replayssm-cache-len=32 \
    --arg mem-fraction-static=0.75 --arg max-running-requests=1 --arg max-mamba-cache-size=64 \
    --arg chunked-prefill-size=8192 --arg max-prefill-tokens=16384 --arg page-size=64 \
    --arg attention-backend=flashinfer --arg fp8-gemm-backend=cutlass \
    --arg flashinfer-mxfp4-moe-precision=default --arg disable-shared-experts-fusion=true \
    --arg cuda-graph-backend-decode=full --arg cuda-graph-max-bs-decode=1 --arg cuda-graph-backend-prefill=disabled \
    --arg reasoning-parser=deepseek-r1 --arg tool-call-parser=qwen25 --arg trust-remote-code=true \
    --arg 'env.SGLANG_ENABLE_FP8_LM_HEAD=1' \
    --request temperature=0 --request seed=42 --request timeout_s=3600 \
    --request 'chat_template_kwargs={"enable_thinking": false}' \
    --out "packets/ling.$W.json" >"logs/ling.$W.packet.log" 2>&1 || { echo "!! packet failed for $W"; cat "logs/ling.$W.packet.log" | tail -5; continue; }
  python3 - "packets/ling.$W.json" "$BUILD" "$NOTES" <<'PY'
import json,sys; p=sys.argv[1]; t=json.load(open(p))
t["engine"]["build"]=sys.argv[2]
t["engine"]["install"]={"method":"source","image":None,"package":"inclusionAI/sglang ling_v3_support_mxfp4_humming, editable"}
t["model"]["served_model_id"]="ling-3.0-flash"
t.setdefault("notes", sys.argv[3])
json.dump(t,open(p,"w"),indent=2)
PY
  uv run atlas-bench run --spec "packets/ling.$W.json" --base-url "$BASE" --out ../results --login "$LOGIN" \
     >"logs/ling.$W.run.log" 2>&1 && touch "logs/ling.$W.done" || echo "!! run failed for $W (see logs/ling.$W.run.log)"
done
echo "ATLAS DONE $(date -Is)"

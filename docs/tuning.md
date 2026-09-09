# Tuning

## Knobs that decide the number

| knob | where | effect |
|---|---|---|
| speculative algorithm | `PROFILE` | DSpark (1.36B drafter, published accept-len 5.29) vs NEXTN (built-in one-layer MTP, ~2.3 observed). The single biggest lever. |
| MoE backend | `PROFILE` | `humming` (Blackwell-tuned) vs `flashinfer_mxfp4` (CUTLASS). |
| LM head | `PROFILE` | online FP8 (`SGLANG_ENABLE_FP8_LM_HEAD=1`) halves LM-head bandwidth per decode step. |
| `--linear-replayssm-cache-len` | `start.sh` | KDA ReplaySSM ring for DSpark. Must be a power of two ≥ 2× the verify window (9) → 32. 16 fails startup. |
| `MEM_FRACTION_STATIC` | `.env` | 0.75 default. Cookbook cell ships 0.85; Ant/NVIDIA say 0.75 already risks a driver lockup and use 0.68. Climb from below. |
| `MAX_RUNNING_REQUESTS` | `.env` | 1. This is a low-latency, single-stream recipe. Raising it shifts every other knob. |

## Levers `tune.sh` can A/B

| variant | flag / env |
|---|---|
| `cutedsl` | `--linear-attn-verify-backend nv_cutedsl` — Blackwell-native ring-writing verify kernel |
| `align` | `--speculative-dspark-align-verify-tokens-to-graph-tier` — spend cuda-graph padding on real verification |
| `mem085` | `MEM_FRACTION_STATIC=0.85` |
| `autotune` | drop `--disable-flashinfer-autotune` (`DROP_AUTOTUNE=1`) — safe once kernels are precompiled |
| `block12` / `block6` | `--speculative-dspark-block-size N` — draft block (gamma); verify window is N+1 |

Not yet exercised: `SGLANG_RAGGED_VERIFY_MODE=cap-accept|compact` with an SPS
cost table from `python -m sglang.benchmark.dspark_sps_profiler` against a
live server — the ragged-verify scheduler that caps verify work to the
expected accept length. Needs `--linear-attn-verify-backend triton|nv_cutedsl`.

## Do not

- Apply the ling-cookbook notebook's `--json-model-override-args` YaRN block.
  The checkpoint is `rope_scaling: null` at a native 262144; forcing YaRN
  factor 2.0 re-scales RoPE the model was never trained with.
  `rope_scaling missing 'factor', defaulting to 1.0` in the log is correct.
- Pass `--reasoning-parser ling3` / `--tool-call-parser ling3` on this branch.
  They exist upstream only. Use `deepseek-r1` / `qwen25`.
- Start a second model while one is resident. See gotchas.

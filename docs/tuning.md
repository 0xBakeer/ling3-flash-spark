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

## Levers `tune.sh` can A/B — and what they were worth

Measured on one DGX Spark, thinking on, greedy, medians over 4-10 runs.
**Every lever was a null result or a loss.** The defaults in this recipe are
already at the plateau; that is the finding, and it is why they are the defaults.

| variant | flag / env | prose | code | verdict |
|---|---|---:|---:|---|
| *(default)* | — | 40.2 | 72.2 | baseline |
| `cutedsl` | `--linear-attn-verify-backend nv_cutedsl` | 40.0 | 72.6 | **null** |
| `align` | `--speculative-dspark-align-verify-tokens-to-graph-tier` | 40.6 | 68.9 | null |
| `mem085` | `MEM_FRACTION_STATIC=0.85` | 40.4 | 68.8 | null, and closer to the memory ceiling |
| `autotune` | drop `--disable-flashinfer-autotune` (`DROP_AUTOTUNE=1`) | 40.4 | 72.9 | null |
| `block6` | `--speculative-dspark-block-size 6` | 43.9 | 70.0 | null |
| `block12` | `--speculative-dspark-block-size 12` | 33.6 | 53.1 | **loss, ~17 tok/s on code** |

`block12` is the one with a visible mechanism: the draft block sets the verify
window (`gamma + 1`), so a wider window makes every verification step more
expensive, and the extra tokens accepted do not pay for it. The drafter's own
checkpoint value of 8 is right, and now measured rather than assumed.

### A warning about the random workload

`cutedsl` first appeared to give **88 tok/s against 51** on the random-8k row —
a 70% win. It was not real. Each variant had been given *different* random
prompts, and on that workload the generated text decides how many draft tokens
are accepted, so different prompts are different experiments. Re-run with
identical prompts (`--seed-salt`), the two arms are within noise. Any lever
result taken from `--workload random` without a shared seed salt is a lottery
ticket, not a measurement.

Not yet exercised: `SGLANG_RAGGED_VERIFY_MODE=cap-accept|compact` with an SPS
cost table from `python -m sglang.benchmark.dspark_sps_profiler` against a
live server — the ragged-verify scheduler that caps verify work to the
expected accept length. Needs `--linear-attn-verify-backend triton|nv_cutedsl`.

## Do not

- Apply the ling-cookbook notebook's `--json-model-override-args` YaRN block.
  The checkpoint is `rope_scaling: null` at a native 262144; forcing YaRN
  factor 2.0 re-scales RoPE the model was never trained with.
  `rope_scaling missing 'factor', defaulting to 1.0` in the log is correct.
- Serve with `--tool-call-parser qwen25`: tool calls come back as text. Use
  `ling3` for both parsers.
- Start a second model while one is resident. See gotchas.

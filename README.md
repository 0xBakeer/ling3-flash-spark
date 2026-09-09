# Ling-3.0-flash (MXFP4) on a single NVIDIA DGX Spark

Serve Ant Group's **Ling-3.0-flash** — 124B total / 5.1B active, hybrid
Kimi-Delta-Attention + gated-MLA MoE, 256K native context — on one DGX Spark
(GB10, sm_121, 121 GiB unified memory) with SGLang.

The point of this recipe is one combination that no published source ships:

> **MXFP4 experts + Humming MoE backend + online FP8 LM head + DSpark speculative decoding.**

The two public DGX Spark sources each have half of it:

| Source | MoE backend | LM head | Spec decode |
|---|---|---|---|
| [SGLang cookbook][cb], `dgx-spark \| mxfp4` cell | `flashinfer_mxfp4` | BF16 | **DSpark** |
| [Ant/NVIDIA notebook][nb] | **`humming`** | **online FP8** | NEXTN (built-in MTP) |
| **this recipe** | **`humming`** | **online FP8** | **DSpark** |

[cb]: https://docs.sglang.io/cookbook/autoregressive/InclusionAI/Ling-3.0-flash
[nb]: https://github.com/inclusionAI/ling-cookbook/blob/main/guide/local-deploy/ling-3.0-flash/dgx-spark-sglang-ling-3-flash-mxfp4-humming.ipynb

All three are selectable, so the comparison can be re-run rather than taken on trust:

```bash
PROFILE=cookbook       ./start.sh    # reproduce the cookbook cell
PROFILE=humming-nextn  ./start.sh    # reproduce the Ant/NVIDIA notebook
PROFILE=humming-dspark ./start.sh    # this recipe (default)
```

## Why these three pieces

**MXFP4, not INT4.** Ant publishes accuracy for every quantisation of this model,
and FP4 beats INT4 on three of four datasets while being the format GB10's
Blackwell tensor cores support natively:

| dataset | BF16 | FP8 | INT4 | **FP4** |
|---|---|---|---|---|
| GPQA-diamond | 84.97 | 84.00 | 83.65 | **83.68** |
| IFBench | 74.49 | 73.40 | 72.20 | **73.87** |
| SciCode | 41.24 | 40.37 | 39.35 | 38.43 |
| ArcPrize | 68.75 | 67.18 | 67.56 | **68.06** |

The checkpoint is mixed, not uniformly 4-bit: routed experts are MXFP4
(e2m1 weights, e8m0 scales, group 32); attention, shared experts, the router and
the embeddings stay FP8 block-128. That is where the quality survives.

**DSpark over NEXTN.** NEXTN uses the model's single built-in MTP layer. DSpark
is a separate 1.36B drafter with a confidence head that picks how many tokens to
propose; its published acceptance length is **5.29** macro-mean (6.57 on
HumanEval, 3.51 on Alpaca) against roughly 2–2.5 for a one-layer MTP. Acceptance
length is the multiplier on decode speed, so this is the single biggest lever.
It is also **lossless** — the target model verifies every drafted token.

**Humming + FP8 LM head.** Humming is the Blackwell-tuned MoE kernel; the online
FP8 LM head halves memory-bandwidth pressure during decode, which is what a
5.1B-active MoE is bottlenecked on.

## Install

Container (image built by the `image` workflow on every `v*` tag, arm64/CUDA 13):

```bash
mkdir -p models && MODE=docker ./run.sh download   # 66 GB + 2.6 GB into ./models
MODE=docker ./run.sh compile                         # precompile kernels, once (needs the GPU)
MODE=docker ./run.sh serve                           # ghcr.io/0xbakeer/ling3-flash-spark
```

Native:

```bash
./setup.sh            # uv venv (py3.11) + SGLang from inclusionAI's humming branch
./download.py         # 66 GB target + 2.6 GB DSpark drafter -> ~/models
./compile-kernel.py   # precompile FlashInfer CUTLASS MXFP4 kernels (~8 min, once)
cp env.example .env   # then edit
./start.sh
```

`compile-kernel.py` is not optional on a single Spark. JIT-compiling those
kernels *while* 63 GB of weights load exhausts the unified pool, and the failure
mode is not an OOM — the driver throws `rmapiLockAcquire` and the box stops
responding. The script runs a probe server only long enough to emit
`build.ninja`, kills it to release the memory, then builds with the box to itself.

## Benchmarking

`bench.py` reports TTFT and TPOT so the numbers line up with the SGLang
cookbook's own cells.

```bash
python3 bench.py --workload prose --osl 1024 --runs 4      # natural generation
python3 bench.py --workload code  --osl 1024 --runs 4
python3 bench.py --workload random --isl 8192 --osl 1024   # the cookbook workload
./sweep.sh                                                 # A/B all three profiles
```

Two traps it exists to avoid:

1. **Speculative decoding packs several tokens into one SSE chunk.** Counting
   chunks under-reports by 3–5x. Always `usage.completion_tokens`.
2. **Repeated filler gets prefix-cached and re-tokenised** at ~6.8 chars/token
   instead of ~4, silently shortening the prompt. Every run builds a fresh
   prompt and verifies its length against the server's own tokenizer.

## Gotchas found building this

- **`--linear-replayssm-cache-len 32` is mandatory with DSpark.** The drafter's
  block size is 8, so the verify window is 9, and the KDA ReplaySSM ring must be
  a power of two at least twice the window. The default 16 fails startup.
- **`ling3` parsers do not exist on the inclusionAI branch.** They are upstream
  (SGLang PR #33561) only. This branch needs `--reasoning-parser deepseek-r1
  --tool-call-parser qwen25`; passing `ling3` aborts startup.
- **flashinfer and flashinfer-cubin versions must match.** The HF card pins
  cubin 0.6.16.post1 against a 0.6.17 flashinfer and papers over it with
  `FLASHINFER_DISABLE_VERSION_CHECK=1`. Installing cubin 0.6.17 is the real fix.
- **Do not apply the notebook's YaRN override.** It sets `rope_scaling` to YaRN
  factor 2.0 over `original_max_position_embeddings 131072`. The checkpoint
  declares `rope_scaling: null` at a native 262144, so that re-scales RoPE the
  model was never trained with. SGLang's `rope_scaling missing 'factor',
  defaulting to 1.0` log line is the *correct* native path. `YARN_OVERRIDE=0`.
- **`--mem-fraction-static` is the dangerous knob.** The cookbook cell ships
  0.85; the Ant/NVIDIA notebook says 0.75 "risks GPU OOM and driver lockups" and
  recommends 0.68. This recipe defaults to 0.75 and climbs from below.
- **One ~60 GB model at a time.** `start.sh` refuses to launch unless
  `MemAvailable` is at least 85 GiB — 63 GB of Ling on top of another resident
  model does not fit in 121 GiB, and the failure mode is a wedged driver, not an
  OOM. `stop.sh` blocks until the memory is actually back for the same reason.
- Weight load is ~340 s cold. Full time to healthy is ~8 min. Don't call it
  failed early.

## Results

All measured rows, raw per-run spreads and the reading of them: [`results/README.md`](results/README.md).
inference-atlas rows are taken with [`atlas/bench_ling.sh`](atlas/bench_ling.sh) against a running server.

## Layout

```
run.sh             dispatcher: setup|download|compile|serve|stop|bench|logs|shell (MODE=docker for compose)
Dockerfile         arm64 CUDA-13 image, SGLang pinned to the branch SHA; weights mounted at /models
setup.sh           build SGLang natively (inclusionAI ling_v3_support_mxfp4_humming)
download.py        fetch target + drafter
compile-kernel.py  precompile FlashInfer CUTLASS MXFP4 kernels
start.sh           serve; PROFILE selects the recipe
stop.sh            graceful stop
bench.py           TTFT / TPOT / decode tok/s
sweep.sh           A/B all profiles on this box
tune.sh            one-lever-at-a-time variants on top of humming-dspark
atlas/             inference-atlas speed-row runner
results/           measured rows (JSON + table)
docs/              install · benchmarking · tuning · gotchas
.env               tunables
```

## Credits

Model, quantised checkpoints and the DSpark drafter: **Ant Group / inclusionAI**.
The DGX Spark Humming + FP8-LM-head tuning and the kernel-precompile trick come
from the Ant Group / NVIDIA `ling-cookbook` notebook (thanks to
[@ly01325](https://github.com/ly01325)); the DSpark flag contract and the
`--linear-replayssm-cache-len` sizing come from the SGLang cookbook. This recipe
combines them and measures the result.

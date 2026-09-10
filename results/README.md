# Results — single NVIDIA DGX Spark

## Headline: the recommended configuration

`PROFILE=humming-dspark`, thinking on (the model's default), recipe defaults,
one request at a time, greedy, 1024 output tokens forced, **10 runs** per
workload with identical prompts across runs (`--seed-salt HEAD`).

| workload | decode tok/s (median) | TPOT | TTFT | run spread |
|---|---:|---:|---:|---|
| code | **72.3** | 13.8 ms | 187 ms | 68.7 – 82.1 |
| prose | **40.9** | 24.5 ms | 191 ms | 38.9 – 58.1 |
| random 8k → 1k | **36.3** | 27.6 ms | 2033 ms | 33.9 – 77.2 |

The spread is not noise in the usual sense — it is DSpark's acceptance length
tracking the content being generated. The same server answers a predictable
continuation at 82 tok/s and an unpredictable one at 69. Report medians over
several prompts; a single favourable run is not a number.

Thinking on is both the more accurate **and** the faster mode on this stack;
see the eval table below and `docs/tuning.md`.

## Same box, same harness — profile comparison

All rows: one request at a time, thinking off, output forced to the full `osl` with `ignore_eos`, medians over the listed runs. Raw per-run TPOTs shown so the spread is visible. Harness: `bench.py` (see `docs/benchmarking.md`). Server: SGLang `0.0.0.dev1+g079d40460`, `--mem-fraction-static 0.75` unless a variant says otherwise.

## Same box, same harness

| profile | workload | isl | osl | runs | TTFT ms | TPOT ms | decode tok/s | per-run TPOT ms | file |
|---|---|---|---|---|---|---|---|---|---|
| `humming-dspark` | random | 8192 | 1024 | 3 | 2111 | 12.29 | **81.4** | [10.6, 20.9, 12.3] | `humming-dspark-8k-1k.json` |
| `humming-dspark/code` | code | ~50 | 1024 | 4 | 179 | 14.62 | **68.4** | [15.2, 13.2, 14.4, 14.9] | `humming-dspark-code.json` |
| `humming-dspark/prose` | prose | ~50 | 1024 | 4 | 185 | 23.07 | **43.4** | [22.8, 23.3, 24.4, 21.9] | `humming-dspark-prose.json` |
| `humming-dspark/random8k` | random | 8192 | 1024 | 3 | 2122 | 13.74 | **72.8** | [9.6, 18.8, 13.7] | `humming-dspark-random8k.json` |
| `tune/base/code` | code | ~50 | 1024 | 4 | 182 | 13.20 | **75.8** | [13.4, 12.9, 13.8, 13.0] | `tune-base-code.json` |
| `tune/base/code-greedy` (greedy) | code | ~50 | 1024 | 3 | 181 | 13.71 | **72.9** | [15.0, 13.0, 13.7] | `tune-base-code-greedy.json` |
| `tune/base/prose` | prose | ~50 | 1024 | 4 | 188 | 23.83 | **42.0** | [24.0, 23.5, 24.0, 23.7] | `tune-base-prose.json` |
| `tune/base/random8k-greedy` (greedy) | random | 8192 | 1024 | 3 | 2051 | 27.87 | **35.9** | [29.2, 24.3, 27.9] | `tune-base-random8k-greedy.json` |
| `cookbook/code` | code | ~50 | 1024 | 4 | 203 | 14.95 | **66.9** | [14.2, 15.7, 15.0, 14.9] | `cookbook-code.json` |
| `cookbook/prose` | prose | ~50 | 1024 | 4 | 208 | 26.45 | **37.8** | [27.1, 25.9, 26.8, 26.1] | `cookbook-prose.json` |
| `cookbook/random8k` | random | 8192 | 1024 | 3 | 2108 | 29.39 | **34.0** | [33.1, 14.9, 29.4] | `cookbook-random8k.json` |
| `cookbook-exact/code` | code | ~50 | 1024 | 4 | 187 | 14.47 | **69.3** | [15.3, 15.3, 12.9, 13.7] | `cookbook-exact-code.json` |
| `cookbook-exact/prose` | prose | ~50 | 1024 | 4 | 188 | 25.08 | **39.9** | [25.6, 24.2, 24.6, 25.9] | `cookbook-exact-prose.json` |
| `cookbook-exact/random8k` | random | 8192 | 1024 | 3 | 2347 | 16.92 | **59.1** | [39.9, 16.2, 16.9] | `cookbook-exact-random8k.json` |
| `cookbook-exact/random8k-greedy` (greedy) | random | 8192 | 1024 | 3 | 188 | 32.99 | **30.3** | [21.2, 33.0, 34.7] | `cookbook-exact-random8k-greedy.json` |
| `humming-nextn/code` | code | ~50 | 1024 | 4 | 178 | 18.11 | **55.2** | [17.9, 17.8, 18.4, 18.4] | `humming-nextn-code.json` |
| `humming-nextn/prose` | prose | ~50 | 1024 | 4 | 178 | 22.23 | **45.0** | [22.1, 22.4, 21.6, 23.0] | `humming-nextn-prose.json` |
| `humming-nextn/random8k` | random | 8192 | 1024 | 3 | 2043 | 19.37 | **51.6** | [18.1, 22.4, 19.4] | `humming-nextn-random8k.json` |

## Profiles

- `humming-dspark` — this recipe: Humming MoE + online FP8 LM head + DSpark (`tune/base` is the same config re-taken after a reboot, with per-label seeds).
- `cookbook` — the SGLang cookbook cell's MoE backend + DSpark on this recipe's shared low-latency flags at 0.75.
- `cookbook-exact` — the SGLang cookbook `dgx-spark | mxfp4 | low-latency | dspark` cell verbatim (`--tp 1 --moe-runner-backend flashinfer_mxfp4` + DSpark flags + `--mem-fraction-static 0.85`, nothing else). Calibration row against its published TTFT 2172 ms / TPOT 9.48 ms.
- `humming-nextn` — the Ant/NVIDIA ling-cookbook DGX Spark notebook: Humming + online FP8 LM head + NEXTN 3-step.

## Reading it

- **code**: DSpark profiles ≈ 68–76 tok/s, NEXTN 55. Acceptance length is the multiplier and the drafter predicts code well.
- **prose**: 40–45 across all profiles — with low acceptance the kernel stack matters less than the draft overhead, and NEXTN's one-layer draft is cheapest.
- **random 8k→1k (the cookbook workload)**: a prompt lottery — identical-length prompts span 9.6–39.9 ms TPOT on the same server because the generated text decides acceptance. Ours reaches the cookbook's published 9.48 ms on its best prompt (9.6 ms); the cookbook cell run verbatim on this box medians 16.9 ms, and ours 13.7 ms on the same prompts.
- Published single-Spark numbers for this model (Ant/NVIDIA guides): llama.cpp Q4_K_M 46.2, vLLM FP4 44.9, SGLang INT4+NEXTN 42.2, SGLang MXFP4 marlin 40.6, vLLM INT4 38.3, Humming+NEXTN notebook 53.8 claimed / 34.9 printed — all short-prompt thinking-on rows, i.e. comparable to the `prose`/`code` rows here, not to `random`.

Lever sweep (`tune.sh`): `base` taken; `cutedsl`, `align`, `mem085`, `autotune` pending.


## Accuracy: what turning thinking off costs

Same weights, same server, same suites; only `--default-chat-template-kwargs`
differs. Published as inference-atlas rows under configs `a3b46e9e` (on) and
`a9e5da49` (off).

| suite | thinking on | thinking off | delta |
|---|---:|---:|---:|
| eval-knowledge-v2 | 1.000 | 1.000 | 0.0 |
| eval-science-v2 | 1.000 | 0.950 | −5.0 |
| eval-math-v2 | 0.993 | 0.821 | −17.2 |
| eval-security-v2 | 0.973 | 0.829 | −14.4 |
| eval-commonsense-v2 | 0.948 | 0.853 | −9.5 |
| eval-reasoning-v2 | 0.950 | 0.643 | **−30.7** |

Ling-3.0-flash spends very few tokens thinking — a couple of dozen on an
arithmetic prompt — so thinking reads as nearly free to switch off for
throughput. It is not. Only pure recall is unaffected. Leave it on: it is the
model's default, it is more accurate, and on this stack it is also faster.

## Tuning

Six levers were A/B'd; none beats these defaults, and one is a clear loss.
Table and the reasoning: `docs/tuning.md`.

# Changelog

Entries are measurement epochs, not code releases: a new entry means the
numbers were re-taken, and says on what.

## 0.1.0 — 2026-09-09

First recipe. Single NVIDIA DGX Spark (GB10, sm_121, 121 GiB unified memory).

- Engine: SGLang `0.0.0.dev1+g079d40460` (inclusionAI
  `ling_v3_support_mxfp4_humming`), torch 2.13.0+cu130, FlashInfer 0.6.17.
- Checkpoint: `inclusionAI/Ling-3.0-flash-fp4` (MXFP4 routed experts, FP8
  everything else, 66 GB) + `inclusionAI/Ling-3.0-flash-dspark` (1.36B BF16).
- Recipe profile `humming-dspark`: Humming MoE + online FP8 LM head + DSpark
  with the KDA ReplaySSM exact fold, 256K native context, `--mem-fraction-static 0.75`.
- Same-box A/B of the two published DGX Spark recipes (`cookbook`,
  `cookbook-exact`, `humming-nextn`) with one harness — see `results/`.
- Harness lessons folded into `bench.py`: `usage.completion_tokens` only
  (speculative decoding packs tokens per chunk), fresh per-run prompts fitted
  against the server tokenizer, per-label seeding so back-to-back rows never
  share prefix-cache blocks, `ignore_eos` so every row decodes the full budget.
- Operational lessons folded into `stop.sh`/`start.sh`: process-group kills,
  a MemAvailable gate on both stop and start, a 45 s grace before a launch is
  judged. All three came from wedging the box once.
- Container: `Dockerfile` + `compose.yaml` + `run.sh` dispatcher; `ghcr.io/0xbakeer/ling3-flash-spark:0.1.0`
  built by GitHub Actions on the tag (arm64). Kernel precompile stays a first-run step (needs the GPU).
- Lever sweep (`tune.sh`: `nv_cutedsl` verify, graph-tier alignment, mem 0.85,
  FlashInfer autotune, block size) started; `base` row taken, remaining
  variants pending.

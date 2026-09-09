# Credits

This recipe combines other people's work and measures the result. The parts
that matter are theirs.

- **Ant Group / inclusionAI** — the Ling-3.0-flash model, the MXFP4 (`-fp4`)
  and INT4 checkpoints with their published per-quant accuracy table, the
  DSpark drafter (`Ling-3.0-flash-dspark`, trained with SpecForge), and the
  `ling_v3_support_mxfp4_humming` SGLang branch that carries the Humming MoE
  adapter, the online FP8 LM head and the draft-shares-target LM-head patch.
  https://huggingface.co/inclusionAI · https://github.com/inclusionAI/sglang
- **Ant Group + NVIDIA, ling-cookbook** — the DGX Spark deployment notebooks,
  in particular the MXFP4 + Humming guide (thanks to
  [@ly01325](https://github.com/ly01325)) whose kernel-precompile trick and
  memory guidance this recipe adopts.
  https://github.com/inclusionAI/ling-cookbook
- **SGLang (LMSYS)** — the engine, DSpark/DFlash speculative decoding, the KDA
  ReplaySSM verify path, and the Ling-3.0-flash cookbook whose `dgx-spark`
  cell defines the DSpark flag contract and the `--linear-replayssm-cache-len`
  sizing. https://docs.sglang.io/cookbook/autoregressive/InclusionAI/Ling-3.0-flash
- **z-lab / DFlash** — the block-diffusion drafting family DSpark extends.
  https://github.com/z-lab/dflash
- **NVIDIA FlashInfer** — the CUTLASS MXFP4 grouped-GEMM MoE kernels compiled
  for sm_121. https://github.com/flashinfer-ai/flashinfer
- **MiaAI Lab** — the `stop.sh` log-archiving pattern and the memory watchdog
  idea come from their single-Spark Qwen3.8-Flash-Next recipe.
  https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark

Measured, assembled and documented by Khaled Bakeer (0xBakeer).

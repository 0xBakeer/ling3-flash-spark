# Install

Target: one NVIDIA DGX Spark (GB10, sm_121, 121 GiB unified memory), DGX OS /
Ubuntu, CUDA 13. Everything runs natively in a uv-managed Python 3.11 venv —
no container, because the JIT kernel build and the 63 GB weight load both
want the whole memory pool and a container only adds a second place to run
out of it.

```bash
./setup.sh
```

`setup.sh` installs uv if missing, creates `.venv` on CPython 3.11, clones
`inclusionAI/sglang` at branch `ling_v3_support_mxfp4_humming` (shallow), and
does an editable install of `sglang/python[all]` with `SGLANG_BUILD_RUST_EXTS=none`
(skips the gRPC/Rust tree, saves ~30 min). It pins `flashinfer-cubin` to the
same version as `flashinfer-python` (0.6.17) — see [gotchas](gotchas.md) for
why the HF model card's pin is wrong.

```bash
./download.py
```

Fetches `inclusionAI/Ling-3.0-flash-fp4` (24 shards, 66 GB) and
`inclusionAI/Ling-3.0-flash-dspark` (2.6 GB) into `~/models/`. Neither repo is
gated. Set `HF_TOKEN` for faster transfers; set `MODEL_DIR` / `DRAFT_DIR` in
`.env` if you keep models elsewhere.

```bash
./compile-kernel.py
```

Mandatory on a single Spark, once. Builds the FlashInfer CUTLASS MXFP4
grouped-GEMM kernels (96 targets, ~8 min) into
`~/.cache/flashinfer/<ver>/121a/cached_ops/fused_moe_120/`. It runs a probe
server only until `build.ninja` appears, kills it, then builds with the box to
itself. Skipping this and letting the first `start.sh` JIT the kernels while
weights load is the documented way to wedge the driver.

```bash
cp env.example .env     # edit if needed
./start.sh              # ~8 min to healthy; ~340 s of that is the weight load
```

Then `curl localhost:30000/v1/models`. Stop with `./stop.sh`, which blocks
until the memory is actually released.

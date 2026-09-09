#!/usr/bin/env python3
"""
compile-kernel.py -- precompile the FlashInfer CUTLASS MXFP4 MoE kernels.

Why this exists: on a single DGX Spark, JIT-compiling these kernels *while*
~60 GB of weights are being loaded exhausts the 121 GiB unified pool. The
failure mode is not an OOM -- the driver throws rmapiLockAcquire and the box
stops responding. So: run a probe server just long enough to emit build.ninja,
kill it to release the memory, then build the kernels with the box to itself.

Adapted from the Ant Group / NVIDIA ling-cookbook DGX Spark notebook
(guide/local-deploy/ling-3.0-flash/dgx-spark-sglang-ling-3-flash-mxfp4-humming).

    ./compile-kernel.py          # ~6-8 minutes, then cached forever
"""
import os, sys, shutil, subprocess, time, pathlib

RECIPE_DIR = pathlib.Path(__file__).resolve().parent
MODEL_DIR = pathlib.Path(os.environ.get("MODEL_DIR", pathlib.Path.home() / "models" / "Ling-3.0-flash-fp4"))
ROOTS = [
    pathlib.Path.home() / ".cache" / "flashinfer",
    pathlib.Path.home() / ".cache" / "sglang" / ".cache" / "flashinfer",
]
PATTERN = "*/121a/cached_ops/fused_moe_120"   # 121a == sm_121, the GB10

def newest(name):
    hits = [p for r in ROOTS for p in r.glob(f"{PATTERN}/{name}")]
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None

def main():
    so = newest("fused_moe_120.so")
    if so:
        print(f"OK: FlashInfer CUTLASS MXFP4 kernel already built, reusing {so}")
        return 0
    if not MODEL_DIR.is_dir():
        print(f"ERROR: model dir not found: {MODEL_DIR}", file=sys.stderr)
        return 1

    print("Probing to generate build rules (this loads weights; ~6-8 min total)...")
    state = {"size": -1, "stable": 0}

    def ninja_ready():
        bn = newest("build.ninja")
        if bn is None:
            return False
        size = bn.stat().st_size
        state["stable"] = state["stable"] + 1 if size == state["size"] else 0
        state["size"] = size
        return state["stable"] >= 3

    env = {
        **os.environ,
        "FLASHINFER_JIT_MAX_JOBS": "1", "MAX_JOBS": "1",
        "NINJA_NUM_JOBS": "1", "CMAKE_BUILD_PARALLEL_LEVEL": "1",
        "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
        "SGLANG_ENABLE_JIT_DEEPGEMM": "0",
        "SGLANG_DSV4_FP4_DEQUANT": "0",
        "SGLANG_FP8_IGNORED_LAYERS": "",
        "SGLANG_ENABLE_FP8_LM_HEAD": "0",
    }
    probe = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", str(MODEL_DIR),
        "--served-model-name", "ling-probe",
        "--trust-remote-code", "--dtype", "bfloat16",
        "--tp-size", "1", "--ep-size", "1",
        "--host", "127.0.0.1", "--port", "30099",
        "--max-running-requests", "1", "--max-mamba-cache-size", "64",
        "--chunked-prefill-size", "8192", "--max-prefill-tokens", "16384",
        "--page-size", "64",
        "--cuda-graph-backend-decode", "full",
        "--cuda-graph-max-bs-decode", "1", "--cuda-graph-bs-decode", "1",
        "--cuda-graph-backend-prefill", "disabled",
        "--random-seed", "308534008",
        "--attention-backend", "flashinfer", "--disable-flashinfer-autotune",
        "--mem-fraction-static", "0.68",
        "--fp8-gemm-backend", "cutlass",
        "--moe-runner-backend", "flashinfer_mxfp4",
        "--flashinfer-mxfp4-moe-precision", "default",
        "--disable-shared-experts-fusion",
        "--enable-fp32-lm-head",
    ]
    log = open(RECIPE_DIR / "logs" / "compile-kernel-probe.log", "wb")
    proc = subprocess.Popen(probe, stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        deadline = time.time() + 3600
        while not ninja_ready():
            time.sleep(5)
            if proc.poll() is not None:
                print("ERROR: probe server exited early; see logs/compile-kernel-probe.log",
                      file=sys.stderr)
                return 1
            if time.time() > deadline:
                print("ERROR: timed out waiting for build.ninja", file=sys.stderr)
                return 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
        # NB: matching on the module path, not a bare "sglang", so this never
        # matches the ssh command line that launched us.
        subprocess.run(["pkill", "-f", "sglang.launch_server"], capture_output=True)
        time.sleep(5)
        log.close()

    bn = newest("build.ninja")
    if bn is None:
        print("ERROR: no build.ninja produced", file=sys.stderr)
        return 1
    build_dir = bn.parent
    ninja_bin = shutil.which("ninja") or "ninja"
    avail_gb = next(int(l.split()[1]) / 1048576 for l in open("/proc/meminfo")
                    if l.startswith("MemAvailable"))
    jobs = max(1, min(os.cpu_count() or 1, int(avail_gb // 10), 8))
    print(f"Memory released ({avail_gb:.0f} GiB free). Building with -j{jobs} in {build_dir}")
    if subprocess.run([ninja_bin, f"-j{jobs}", "-C", str(build_dir)]).returncode != 0:
        print("ERROR: ninja build failed", file=sys.stderr)
        return 1
    print("OK: FlashInfer CUTLASS MXFP4 kernels compiled")
    return 0

if __name__ == "__main__":
    sys.exit(main())

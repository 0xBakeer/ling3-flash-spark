# Gotchas

Every one of these was hit for real while building this.

## The box wedges instead of OOMing
Exhaust the 121 GiB unified pool and there is no OOM message and no log line:
the driver throws `rmapiLockAcquire`, sshd stops completing handshakes, and
the only way back is the power button. Three separate paths led there:

1. JIT-compiling FlashInfer kernels while 63 GB of weights load →
   `compile-kernel.py` first, always.
2. Starting a second server before the first one's memory is released →
   `stop.sh` blocks until `MemAvailable ≥ 90 GiB`; `start.sh` refuses below 85.
3. A sweep that judged a launch "failed" before python had exec'd and started
   the next profile on top of it → 45 s grace before any liveness check.

## Killing SGLang is more than one `pkill`
It forks a scheduler, a detokeniser and workers; several do not carry the
model path or even `launch_server` on argv. `stop.sh` kills process *groups*,
launches are wrapped in `setsid` so a group-kill can never reach the caller,
and it compares against its own PGID — comparing against `$$` (a PID) once
killed the sweep that called it.

Also: `pkill -f <pattern>` from an ssh one-liner whose command line contains
the pattern kills the session itself. Use `pkill -f "patter[n]"`.

## `--linear-replayssm-cache-len 32`
DSpark's block size is 8, so the verify window is 9. The KDA ReplaySSM ring
must be a power of two at least twice that. The default 16 fails startup
validation with a clear message; 32 is the smallest that passes.

## Parsers: `ling3`, both of them
Ling-3.0-flash's tool-call format is GLM-4.5 XML:
`<tool_call>get_weather<arg_key>city</arg_key><arg_value>Hangzhou</arg_value></tool_call>`,
sometimes without the newline after the name. `--tool-call-parser qwen25`
expects `<tool_call>{json}</tool_call>` and never matches, so the call is
returned as chat text (0.1.0 shipped that). `ling3` is a `Glm4MoeDetector`
subclass built for exactly this layout, and the `ling3` reasoning parser
pairs with it. Both are on the inclusionAI branch — registered in
`function_call_parser.py` / `reasoning_parser.py`, not in `server_args.py`.
`enable_thinking: false` can still route a few tokens to `reasoning_content`;
read both fields.

## flashinfer / flashinfer-cubin version lock
The HF card pins `flashinfer-cubin==0.6.16.post1` against a 0.6.17
`flashinfer-python` and papers over it with `FLASHINFER_DISABLE_VERSION_CHECK=1`.
Install the matching cubin (0.6.17). The mismatch aborts the probe server.

## YaRN override
Don't. See [tuning](tuning.md).

## Cold start is ~8 minutes
~340 s is the weight load alone (66 GB from NVMe), then the drafter, then
cuda-graph capture. Nothing is wrong at minute 5.

## Random-input benchmarks are a prompt lottery
Same server, same length, two prompts: 9.6 ms vs 18.8 ms TPOT. The generated
text decides acceptance. Report medians over several prompts and keep the
raw runs. A published single number on this workload can be reproduced with
the right prompt and missed with the wrong one.

## Streaming chunk counts lie under speculative decoding
Several accepted tokens arrive per SSE chunk. Use `usage.completion_tokens`.

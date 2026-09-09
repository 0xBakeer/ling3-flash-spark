# Benchmarking

`bench.py` measures one server on three workloads and reports the two numbers
the SGLang cookbook reports — TTFT and TPOT — plus decode tok/s derived from
them, so a row here lines up with a cookbook cell.

```bash
python3 bench.py --workload prose  --osl 1024 --runs 4          # short prompt, natural generation
python3 bench.py --workload code   --osl 1024 --runs 4
python3 bench.py --workload random --isl 8192 --osl 1024 --runs 3   # the cookbook's workload
python3 bench.py ... --temperature 0                              # greedy row
./sweep.sh      # A/B every PROFILE on this box
./tune.sh       # one-lever-at-a-time variants on top of humming-dspark
```

## Definitions

- **TTFT** — wall time from request send to the first content or
  reasoning delta.
- **TPOT** — `(total − TTFT) / (completion_tokens − 1)`.
- **decode tok/s** — `1000 / TPOT`. Medians over runs; every run's raw
  numbers are kept in the JSON.

## Four things the harness refuses to get wrong

1. **Token counts come from `usage.completion_tokens`, never from chunk
   counts.** Speculative decoding emits several accepted tokens per SSE chunk;
   counting chunks under-reports decode speed by 3–5×. `stream_options.include_usage`
   is on for every request and a run without usage is an error, not a guess.
2. **Every run gets a fresh prompt, and its length is verified.** Repeated
   filler is prefix-cached (fake TTFT) and tokenises at ~6.8 chars/token instead
   of ~4 (a prompt half the intended length). `random` prompts are unique
   numbered words, fitted by ratio iteration until the *server's own tokenizer*
   reports the target length; a bounded bisect saturates and silently returns
   a 2× prompt — that bug shipped for one run here.
3. **Seeds include the label.** Two rows taken back-to-back on one server
   (e.g. sampled then greedy) must not share prompts, or the second row's TTFT
   is served from the prefix cache.
4. **`ignore_eos` forces the full output budget**, so TPOT is a decode-only
   number over exactly `osl` tokens on every row.

## What the workloads mean

- `prose` / `code`: ~50-token prompts, 1024 generated tokens. This is the
  everyday number, and where a speculative drafter shows its acceptance
  profile — the DSpark card lists 6.57 accepted tokens/step on HumanEval and
  3.51 on Alpaca, and `code` vs `prose` here tracks that ratio.
- `random`: 8192 random-word tokens in, 1024 out, concurrency 1 — the cookbook
  cell's workload. On garbage input the *generated* text decides acceptance,
  which makes this workload a prompt lottery: two prompts of identical length
  gave 9.6 ms and 18.8 ms TPOT on the same server. Report the median and keep
  the raw runs; a single favorable run is not a number.

Speculative decoding is lossless (the target verifies every draft token), so
these workloads measure speed only. Quality is Ant's per-quant table on the
model card — reproduced in the README — not something a speed harness can see.

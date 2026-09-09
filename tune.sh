#!/usr/bin/env bash
# tune.sh -- one-lever-at-a-time A/B on top of PROFILE=humming-dspark.
# Each variant restarts the server with ONE change, then runs the same three
# workloads bench.py uses everywhere, so every row is comparable to the sweep.
#   ./tune.sh                       # all variants
#   VARIANTS="cutedsl mem085" ./tune.sh
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source .venv/bin/activate
mkdir -p bench/results

declare -A VFLAGS VENV
VFLAGS[base]=""                                                        ; VENV[base]=""
VFLAGS[cutedsl]="--linear-attn-verify-backend nv_cutedsl"              ; VENV[cutedsl]=""
VFLAGS[align]="--speculative-dspark-align-verify-tokens-to-graph-tier" ; VENV[align]=""
VFLAGS[autotune]=""                                                    ; VENV[autotune]="TUNE_AUTOTUNE=1"
VFLAGS[mem085]=""                                                      ; VENV[mem085]="TUNE_MEM=0.85"
VFLAGS[block12]="--speculative-dspark-block-size 12"                   ; VENV[block12]=""
VFLAGS[block6]="--speculative-dspark-block-size 6"                     ; VENV[block6]=""

VARIANTS="${VARIANTS:-base cutedsl align autotune mem085 block12 block6}"

for V in $VARIANTS; do
    echo "############################################################"
    echo "### variant $V  flags=[${VFLAGS[$V]}] env=[${VENV[$V]}]"
    if ! MIN_FREE_GIB=90 ./stop.sh; then echo "### ABORT: memory not released"; exit 1; fi
    MEM=0.75; AUTOTUNE_FLAG=""
    for kv in ${VENV[$V]}; do
        case "$kv" in
            TUNE_MEM=*) MEM="${kv#TUNE_MEM=}" ;;
            TUNE_AUTOTUNE=1) AUTOTUNE_FLAG="__drop_autotune__" ;;
        esac
    done
    if [[ "$AUTOTUNE_FLAG" == "__drop_autotune__" ]]; then
        # start.sh always passes --disable-flashinfer-autotune; give it a
        # one-shot override to omit it.
        export DROP_AUTOTUNE=1
    else
        unset DROP_AUTOTUNE
    fi
    PROFILE=humming-dspark MEM_FRACTION_STATIC="$MEM" EXTRA_FLAGS="${VFLAGS[$V]}" \
        setsid nohup ./start.sh > "logs/tune-$V.log" 2>&1 &
    sleep 45
    up=0
    for i in $(seq 1 80); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:30000/health 2>/dev/null)" = "200" ] && { up=1; echo "### ready after ~$((45+i*15))s"; break; }
        pgrep -f 'sglang\.launch_server' >/dev/null || { echo "### FAILED to start"; break; }
        sleep 15
    done
    if [[ $up -ne 1 ]]; then
        echo "### $V did not come up"; grep -aE "Traceback|Error|assert" "logs/tune-$V.log" | grep -av -i warn | tail -4; continue
    fi
    for W in prose code; do
        timeout 1800 python3 bench.py --label "tune/$V/$W" --workload "$W" --osl 1024 --runs 4 --warmup 1 \
            --out "bench/results/tune-$V-$W.json" 2>&1 | grep -E "MEDIAN|isl="
    done
    # Greedy on the cookbook workload: with temperature 0.6 on random input the
    # sampled trajectory, not the kernel, decides acceptance (seen 9.6 vs 18.8 ms
    # TPOT on two prompts of the same length). Temperature 0 removes that noise.
    timeout 2400 python3 bench.py --label "tune/$V/random8k-greedy" --workload random --isl 8192 --osl 1024 \
        --runs 3 --warmup 1 --temperature 0 --out "bench/results/tune-$V-random8k-greedy.json" 2>&1 | grep -E "MEDIAN|isl="
    timeout 1800 python3 bench.py --label "tune/$V/code-greedy" --workload code --osl 1024 \
        --runs 3 --warmup 1 --temperature 0 --out "bench/results/tune-$V-code-greedy.json" 2>&1 | grep -E "MEDIAN"
    echo "### accept len ($V):"; grep -aoE "accept len: [0-9.]+" "logs/tune-$V.log" | tail -3 | tr '\n' ' '; echo
done
echo "TUNE COMPLETE"

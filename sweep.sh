#!/usr/bin/env bash
# sweep.sh -- A/B the three profiles on THIS box with THIS harness.
# Comparing our numbers to someone else's published numbers is not a comparison;
# only a same-box, same-script run settles which recipe is faster.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source .venv/bin/activate
mkdir -p bench/results

PROFILES="${PROFILES:-cookbook humming-nextn}"
WORKLOADS="${WORKLOADS:-prose code}"

for P in $PROFILES; do
    echo "############################################################"
    echo "### profile $P -- restarting"
    # Hard gate: the next 63 GB load must not begin until the previous server's
    # memory is genuinely back. This is what wedged the box on 2026-09-09.
    if ! MIN_FREE_GIB=90 ./stop.sh; then
        echo "### ABORT: memory not released after stopping the previous profile"
        free -g | head -2
        exit 1
    fi
    PROFILE="$P" setsid nohup ./start.sh > "logs/serve-$P.log" 2>&1 &
    sleep 45                                   # let python exec before judging it
    for i in $(seq 1 80); do
        code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:30000/health 2>/dev/null)
        [ "$code" = "200" ] && { echo "### ready after ~$((45 + i*15))s"; break; }
        if ! pgrep -f 'sglang\.launch_server' >/dev/null; then
            echo "### FAILED to start (no launch_server process after $((45 + i*15))s)"
            break
        fi
        sleep 15
    done
    if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:30000/health 2>/dev/null)" != "200" ]; then
        echo "### $P did not come up; see logs/serve-$P.log"
        grep -aE "Traceback|ValueError|RuntimeError|AssertionError|Error:" "logs/serve-$P.log" | tail -5
        continue
    fi
    for W in $WORKLOADS; do
        timeout 1800 python3 bench.py --label "$P/$W" --workload "$W" --osl 1024 \
            --runs 4 --warmup 1 --out "bench/results/$P-$W.json" 2>&1 | tail -8
        echo
    done
    timeout 2400 python3 bench.py --label "$P/random8k" --workload random --isl 8192 --osl 1024 \
        --runs 3 --warmup 1 --out "bench/results/$P-random8k.json" 2>&1 | tail -7
    echo "### accept len samples ($P):"
    grep -aoE "accept len: [0-9.]+" "logs/serve-$P.log" | tail -4
    echo
done
echo "SWEEP COMPLETE"

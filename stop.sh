#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# stop.sh -- stop the SGLang Ling server AND every child it spawned.
#
# Why this is more than one pkill: SGLang forks a scheduler, a detokeniser and
# worker processes. Several of them do NOT carry the model path (or even
# "launch_server") on their command line, so a pattern match on the launch
# command leaves them alive -- still holding the full ~63 GB of weights. Start
# the next server on top of that and a 121 GiB box tries to hold 126 GB, which
# wedges the driver with no OOM and no logs. So: kill the process group, then
# sweep by name, then WAIT for the memory to actually come back.
#
#   ./stop.sh              # graceful, waits for memory to be released
#   ./stop.sh --force      # skip straight to SIGKILL
#   MIN_FREE_GIB=90 ./stop.sh
# ---------------------------------------------------------------------------
set -uo pipefail

FORCE=false
[[ "${1:-}" == "-f" || "${1:-}" == "--force" ]] && FORCE=true
MIN_FREE_GIB="${MIN_FREE_GIB:-80}"
# Our own process group: never signal it, or we kill our caller.
OWN_PGID=$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')
SELF=$$

avail_gib() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }

# Every sglang process except this script and its own children. Matching on
# "sglang" alone would match an ssh command line that merely mentions it, so we
# always exclude our own pid and our parent.
sglang_pids() {
    {
        pgrep -f 'sglang\.launch_server' 2>/dev/null || true
        pgrep -f 'sglang::'              2>/dev/null || true
        pgrep -f 'sglang\.srt'           2>/dev/null || true
    } | sort -u | grep -vx "$SELF" | grep -vx "$PPID" || true
}

pids=$(sglang_pids)
if [[ -z "$pids" ]]; then
    echo "no SGLang server running (MemAvailable $(avail_gib) GiB)"
    exit 0
fi

echo "stopping SGLang pids: $(echo "$pids" | tr '\n' ' ')"

# Kill the whole process group of each launcher: that reaches the children whose
# command lines we cannot match.
if [[ "$FORCE" == false ]]; then
    for p in $pids; do
        pgid=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' ')
        if [[ -n "$pgid" && "$pgid" != "$OWN_PGID" ]]; then
            kill -TERM -- "-$pgid" 2>/dev/null || kill -TERM "$p" 2>/dev/null || true
        else
            kill -TERM "$p" 2>/dev/null || true
        fi
    done
    for _ in $(seq 1 45); do
        [[ -z "$(sglang_pids)" ]] && break
        sleep 2
    done
fi

# Anything still up gets SIGKILL, group first.
remaining=$(sglang_pids)
if [[ -n "$remaining" ]]; then
    echo "escalating to SIGKILL: $(echo "$remaining" | tr '\n' ' ')"
    for p in $remaining; do
        pgid=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' ')
        if [[ -n "$pgid" && "$pgid" != "$OWN_PGID" ]]; then
            kill -KILL -- "-$pgid" 2>/dev/null || kill -KILL "$p" 2>/dev/null || true
        else
            kill -KILL "$p" 2>/dev/null || true
        fi
    done
    sleep 5
fi

if [[ -n "$(sglang_pids)" ]]; then
    echo "ERROR: SGLang processes survived SIGKILL: $(sglang_pids | tr '\n' ' ')" >&2
    exit 1
fi

# The kernel reclaims 63 GB of page-cache-backed weights lazily. Starting the
# next server before that lands is exactly how the box gets wedged, so block.
echo -n "waiting for memory to be released (target >= ${MIN_FREE_GIB} GiB): "
for _ in $(seq 1 60); do
    a=$(avail_gib)
    if (( a >= MIN_FREE_GIB )); then
        echo "OK, ${a} GiB available"
        exit 0
    fi
    sleep 3
done
echo "WARNING: only $(avail_gib) GiB available after 3 min -- not starting anything else is advised" >&2
exit 1

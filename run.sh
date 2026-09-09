#!/usr/bin/env bash
# run.sh -- dispatcher. Native (uv venv) by default; MODE=docker uses compose.
#   ./run.sh setup | download | compile | serve | stop | bench [args] | logs | shell
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${MODE:-native}"
cmd="${1:-serve}"; shift || true
if [[ "$MODE" == "docker" ]]; then
  case "$cmd" in
    setup)    docker compose build ;;
    download) docker compose run --rm --entrypoint /app/run.sh ling download ;;
    compile)  docker compose run --rm --entrypoint /app/run.sh ling compile ;;
    serve)    docker compose up -d ling ;;
    stop)     docker compose stop -t 60 ling; docker compose rm -f ling ;;
    bench)    docker compose exec ling /app/run.sh bench "$@" ;;
    logs)     docker compose logs -f ling ;;
    shell)    docker compose exec ling bash ;;
    *) echo "unknown: $cmd" >&2; exit 2 ;;
  esac
  exit 0
fi
case "$cmd" in
  setup)    ./setup.sh ;;
  download) [[ -f .venv/bin/activate ]] && source .venv/bin/activate; ./download.py ;;
  compile)  source .venv/bin/activate; ./compile-kernel.py ;;
  serve)    exec ./start.sh "$@" ;;
  stop)     ./stop.sh "$@" ;;
  bench)    source .venv/bin/activate; python3 bench.py "$@" ;;
  logs)     tail -f logs/*.log ;;
  shell)    source .venv/bin/activate; exec bash ;;
  *) echo "unknown: $cmd" >&2; exit 2 ;;
esac

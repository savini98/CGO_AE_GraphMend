#!/usr/bin/env bash
#
# Container entry point: one command per claim.
#
#   docker run --gpus all graphmend            # C1 then C2
#   docker run --gpus all graphmend c1         # C1 only
#   docker run --gpus all graphmend c2         # C2 only
#   docker run --gpus all graphmend c3         # C3, the 10-model sample
#   docker run --gpus all graphmend c3 --full  # C3, every row
#   docker run --gpus all graphmend bash       # a shell in the image
#
# Anything after the selector is forwarded, so a subset or --offline still
# works: `graphmend c1 t5-small`, `graphmend c1 --offline`.
#
# A first argument that is not a selector is treated as an argument to the
# default, so `graphmend --offline` and `graphmend t5-small` behave the way
# someone would expect rather than erroring on a name they did not know was
# reserved.

set -uo pipefail

ART_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    sed -n '3,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

sel="${1:-}"
case "$sel" in
    c1)   shift; exec bash "$ART_DIR/run_break_analysis.sh" --c1 "$@" ;;
    c2)   shift; exec bash "$ART_DIR/run_break_analysis.sh" --c2 "$@" ;;
    c1c2|all) shift; exec bash "$ART_DIR/run_break_analysis.sh" "$@" ;;
    c3)   shift
          LAT="$ART_DIR/run_latency_analysis.py"
          if [ ! -f "$LAT" ]; then
              echo "C3 is not present in this image ($LAT is missing)." >&2
              echo "C1 and C2 are available; see 'help'." >&2
              exit 2
          fi
          exec "${PYTHON:-python3}" "$LAT" "$@" ;;
    bash|sh) shift; exec bash "$@" ;;
    -h|--help|help) usage; exit 0 ;;
    *)    exec bash "$ART_DIR/run_break_analysis.sh" "$@" ;;
esac

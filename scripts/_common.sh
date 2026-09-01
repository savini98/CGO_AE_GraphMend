# Shared preamble for the per-claim scripts. Sourced, not executed.
#
# Every script in this directory maps one command to one claim in the paper,
# prints the claim it is checking and what to expect, and leaves the numbers to
# the harness. They are thin on purpose: the measurement lives in paper_eval/
# and artifact/gpu/, so a reviewer reading a script sees which claim it serves
# without reading a benchmark.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="$REPO/jaseci/jac:$REPO${PYTHONPATH:+:$PYTHONPATH}"

banner() {
    printf '\n%s\n' "=================================================================="
    printf '%s\n' "$1"
    printf '%s\n' "  paper:    $2"
    printf '%s\n' "  expect:   $3"
    printf '%s\n\n' "=================================================================="
}

need_setup() {
    if [ ! -d "$REPO/jaseci/jac/jaclang/compiler/passes/graphmend" ]; then
        echo "GraphMend passes are missing. Run:  bash scripts/setup.sh" >&2
        exit 2
    fi
}

need_cuda() {
    "$PYTHON" - <<'PY' || exit 2
import sys
try:
    import torch
except Exception as exc:
    sys.exit(f"torch is not importable: {exc}")
if not torch.cuda.is_available():
    sys.exit(
        "no CUDA device visible. This script measures latency and needs a GPU.\n"
        "The break-elimination, correctness and full-graph claims do not:\n"
        "  bash artifact/run_break_analysis.sh --c1 t5-small"
    )
PY
}

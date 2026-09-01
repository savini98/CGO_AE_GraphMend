#!/usr/bin/env bash
#
# GraphMend CGO 2027 artifact: the break-elimination claim, end to end.
#
#   bash artifact/run_break_analysis.sh                 # every model
#   bash artifact/run_break_analysis.sh t5-small biogpt  # a subset
#   bash artifact/run_break_analysis.sh --offline        # no downloads
#   bash artifact/run_break_analysis.sh --c1             # C1 only
#   bash artifact/run_break_analysis.sh --c2             # C2 only
#
# THE CLAIM: GraphMend eliminates graph breaks, and the transformed program
# produces the same output as the original.
#
# Every model is run twice, GraphMend off then on, through `jac run` with a
# jac.toml differing only in `graphmend_claim_imports`. So what is measured is
# the compiler transforming imported model code, not a hand-edited model file.
# For each row this reports the breaks found, how many were eliminated, and
# whether the two arms produced a bit-identical output.
#
# Correctness is the load-bearing half: eliminating a graph break while
# altering the result is not a fix, so a row whose arms disagree fails the run
# rather than being counted as a successful reduction.
#
# Exit status is 0 only if every row ran and no row changed its output.
#
# This wrapper exists because the measurement is easy to invoke wrongly. It
# checks the three things that silently produce a meaningless result, then runs
# the analysis from the directory it expects with the PYTHONPATH it needs.
#
# Environment overrides:
#   PYTHON=...   interpreter to use (default: python3)

set -uo pipefail

ART_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$ART_DIR/.." && pwd)"
# The toolchain is the pinned jaseci submodule plus patches/graphmend.patch;
# the harness sits at the repository root. Both go on PYTHONPATH.
JAC="$REPO/jaseci/jac"
HARNESS="$REPO"
PYTHON="${PYTHON:-python3}"

# --c1 / --c2 select one claim; with neither, both run. Everything else (model
# names, --offline) is forwarded to whichever steps run.
RUN_C1=0
RUN_C2=0
SELECTED=0
ARGS=()
for _a in "$@"; do
    case "$_a" in
        --c1) RUN_C1=1; SELECTED=1 ;;
        --c2) RUN_C2=1; SELECTED=1 ;;
        *)    ARGS+=("$_a") ;;
    esac
done
if [ "$SELECTED" = "0" ]; then RUN_C1=1; RUN_C2=1; fi
set -- ${ARGS[@]+"${ARGS[@]}"}

rule() { printf '\n%s\n' "------------------------------------------------------------------"; }

rule
echo "STEP 0  environment"
rule

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "  no such interpreter: $PYTHON"
    echo "  set PYTHON=/path/to/python3 (3.13 is what the results were measured on)."
    exit 2
fi

if [ ! -d "$JAC/jaclang/compiler/passes/graphmend" ]; then
    echo "  no GraphMend passes under $JAC"
    echo "  the toolchain is a pinned submodule plus patches/graphmend.patch,"
    echo "  and one or both is missing. Run:  bash scripts/setup.sh"
    exit 2
fi
if [ ! -f "$HARNESS/paper_eval/run_eval.py" ]; then
    echo "  cannot find $HARNESS/paper_eval/run_eval.py"
    echo "  run this from a checkout of the artifact, as"
    echo "  'bash artifact/run_break_analysis.sh'."
    exit 2
fi

echo "  repo        $REPO"
echo "  interpreter $("$PYTHON" -c 'import sys;print(sys.executable)') ($("$PYTHON" -c 'import platform;print(platform.python_version())'))"

# importlib.metadata reads installed distribution metadata; it does not import
# torch, so this stays cheap even where torch takes seconds to load.
VERSIONS="$("$PYTHON" - <<'PY'
import importlib.metadata as md
for pkg in ("torch", "transformers", "numpy"):
    try:
        print(f"{pkg} {md.version(pkg)}")
    except Exception:
        print(f"{pkg} MISSING")
PY
)"
echo "$VERSIONS" | while read -r name ver; do printf '  %-12s %s\n' "$name" "$ver"; done

TORCH_V="$(printf '%s\n' "$VERSIONS" | awk '$1=="torch"{print $2}')"
TRANSFORMERS_V="$(printf '%s\n' "$VERSIONS" | awk '$1=="transformers"{print $2}')"

if [ "$TORCH_V" = "MISSING" ] || [ "$TRANSFORMERS_V" = "MISSING" ]; then
    echo
    echo "  torch and transformers are required. See artifact/README.md, or use"
    echo "  the container: docker build -f artifact/Dockerfile -t graphmend ."
    exit 2
fi

# The local version segment is stripped before comparing: a wheel from the
# PyTorch index reports its build there ("2.12.1+cpu", "2.12.1+cu126"), and
# comparing the whole string would warn on every containerised run.
[ "${TORCH_V%%+*}" = "2.12.1" ] || echo "  NOTE: torch $TORCH_V, results were measured on 2.12.1"
[ "$TRANSFORMERS_V" = "4.52.4" ] || echo "  NOTE: transformers $TRANSFORMERS_V, results were measured on 4.52.4"

# The typeshed stdlib stubs are gitignored, and type inference is on the
# critical path of every Jac compilation. Without them the toolchain imports
# fine, caches modules fine, and then dies on the first real compile with
# TypeshedUnavailableError, which reads like a harness bug rather than a
# missing fetch. Check it here instead.
if [ ! -d "$JAC/jaclang/vendor/typeshed/stdlib" ]; then
    echo
    echo "  the vendored typeshed stdlib stubs are missing, so no Jac"
    echo "  compilation can run. Materialize them with:"
    echo "      $PYTHON artifact/fetch_typeshed.py jac"
    exit 2
fi
echo "  typeshed    present"

STATUS=0
if [ "$RUN_C1" = "1" ]; then
rule
echo "C1  graph breaks repaired, model behaviour preserved"
rule
echo "  each row compiles its model twice, GraphMend off then on, so this is"
echo "  slow on a cold cache. Each row prints as it finishes."
echo

# PYTHONUNBUFFERED so a long run is not silent, and PYTHONPATH so the vendored
# toolchain is importable without a pip install.
PYTHONUNBUFFERED=1 PYTHONPATH="$JAC:$HARNESS${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" "$ART_DIR/verify_break_elimination.py" "$@"
STATUS=$?
fi

# ---------------------------------------------------------------------------
# C2 -- full-graph capture (paper Section 5.6).
#
# This is what C1 buys. vLLM and SGLang require
# torch.compile(fullgraph=True), which a single graph break defeats, so a model
# whose breaks are gone should now capture where it previously could not. The
# check is asymmetric on purpose: the untransformed arm must FAIL and the
# transformed arm must SUCCEED, so a both-pass result is a failure rather than
# a win. backend="eager" isolates Dynamo's capture from backend compilation,
# which makes it deterministic and keeps it off the GPU.
# ---------------------------------------------------------------------------
FG_STATUS=0
if [ "$RUN_C2" = "1" ]; then
rule
echo "C2  full-graph capture for models whose breaks are fully repaired"
rule
echo "  torch.compile(fullgraph=True) should fail before the transform and"
echo "  succeed after it, on every row."
echo

# Forward the selection verbatim. run_fullgraph understands both model names
# and --offline, so `run_break_analysis.sh t5-small` checks capture for
# t5-small alone rather than for all 21 rows, and --offline drops the rows that
# download from this step as well as from the one above.
PYTHONUNBUFFERED=1 PYTHONPATH="$JAC:$HARNESS${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" -m paper_eval.run_fullgraph "$@"
FG_STATUS=$?
fi
[ "$STATUS" = "0" ] || FG_STATUS="$STATUS"

rule
echo "SUMMARY"
rule
if [ "$FG_STATUS" = "0" ]; then
    [ "$RUN_C1" = "1" ] && echo "C1: graph breaks were repaired and every compared output was identical."
    [ "$RUN_C2" = "1" ] && echo "C2: every fully repaired model captured as a full graph where the"
    [ "$RUN_C2" = "1" ] && echo "    original could not."
else
    echo "NOT established by this run: a row failed to run or changed its"
    echo "output. The table above says which."
    echo "artifact/README.md has a troubleshooting section; the two gotchas at"
    echo "the top of it account for most unexpected rows."
fi
exit "$FG_STATUS"

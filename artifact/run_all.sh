#!/usr/bin/env bash
#
# GraphMend CGO 2027 artifact: the kick-the-tires path.
#
#   bash artifact/run_all.sh              # default: 4 rule suites + 5 models
#   bash artifact/run_all.sh --quick      # 4 rule suites + 2 models
#   bash artifact/run_all.sh --suites     # 4 rule suites only, no model rows
#
# CPU only. No GPU. No network. No model weights are downloaded: every model in
# the default set is built from a small random-weight config, because graph
# breaks are structural (they are code paths) and do not depend on weights.
#
# Exit status is 0 only if every check passed.
#
# Environment overrides:
#   PYTHON=...           interpreter to use (default: python3)
#   JAC_TEST_JOBS=N      test worker count (default: 1, see note below)

set -uo pipefail

ART_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$ART_DIR/.." && pwd)"
JAC="$REPO/jac"
PYTHON="${PYTHON:-python3}"
LOG_DIR="${TMPDIR:-/tmp}/graphmend-artifact-$$"
mkdir -p "$LOG_DIR"

# The repo root jac.toml pins `test_jobs = "8"` for the full compiler suite,
# where workers were measured at ~5 GB each. The four suites below are small,
# so default to one process and let a reviewer with headroom raise it.
export JAC_TEST_JOBS="${JAC_TEST_JOBS:-1}"

# `jac run` and `jac test` here are the in-repo compiler source, reached with
# PYTHONPATH rather than a pip install. The toolchain declares no runtime PyPI
# dependencies, so there is nothing else to install for it.
export PYTHONPATH="$JAC${PYTHONPATH:+:$PYTHONPATH}"

RESULTS=()
FAILED=0

pass() { RESULTS+=("PASS  $1"); printf '  PASS  %s\n' "$1"; }
fail() { RESULTS+=("FAIL  $1"); FAILED=1; printf '  FAIL  %s\n' "$1"; }
rule() { printf '\n%s\n' "------------------------------------------------------------------"; }

# ---------------------------------------------------------------------------
# Step 0: environment
# ---------------------------------------------------------------------------
rule
echo "STEP 0  environment"
rule

if [ ! -f "$JAC/paper_eval/run_eval.py" ]; then
    echo "  cannot find $JAC/paper_eval/run_eval.py"
    echo "  run this script from a checkout of the artifact branch, as"
    echo "  'bash artifact/run_all.sh'."
    exit 2
fi

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "  no such interpreter: $PYTHON"
    echo "  set PYTHON=/path/to/python3 (3.13 is what the results were measured on)."
    exit 2
fi

echo "  repo        $REPO"
echo "  interpreter $($PYTHON -c 'import sys;print(sys.executable)') ($($PYTHON -c 'import platform;print(platform.python_version())'))"

# importlib.metadata reads the installed distribution metadata; it does not
# import torch, so this stays cheap even where torch takes seconds to load.
VERSIONS="$($PYTHON - <<'PY'
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
    echo "  the container: docker build -f artifact/Dockerfile.cpu -t graphmend-cpu ."
    exit 2
fi

# The measured results in artifact/README.md were taken on exactly these two
# versions. Anything else may still work and is not blocked, but the numbers
# below are only claimed for this pair.
#
# The local version segment is stripped before comparing. A wheel from the
# PyTorch index reports its build in that segment -- "2.12.1+cpu", "2.12.1+cu126"
# -- and comparing the whole string would warn that the correct torch is the
# wrong one on every containerised run.
[ "${TORCH_V%%+*}" = "2.12.1" ] || echo "  NOTE: torch $TORCH_V, results were measured on 2.12.1"
[ "$TRANSFORMERS_V" = "4.52.4" ] || echo "  NOTE: transformers $TRANSFORMERS_V, results were measured on 4.52.4"

MODE="default"
case "${1:-}" in
    --quick)  MODE="quick" ;;
    --suites) MODE="suites" ;;
    "")       ;;
    *) echo "  unknown option: $1 (expected --quick, --suites, or nothing)"; exit 2 ;;
esac
echo "  mode        $MODE"
echo "  logs        $LOG_DIR"

# ---------------------------------------------------------------------------
# Step 1: rule-level graph-count suites
#
# These are the claim at its smallest: a region that hands TorchDynamo two or
# more FX graphs untransformed hands it exactly one after the rewrite. One
# suite per rule, plus the import-claiming path. They run a real counting
# Dynamo backend, so they are the same kind of measurement as the model rows,
# on fixtures rather than on Hugging Face code.
#
# Each test skips itself when torch is missing, and a fully skipped session
# still exits 0, so this step fails on any skip as well as on any failure.
# ---------------------------------------------------------------------------
rule
echo "STEP 1  rule-level graph-count suites (expect 18 passed, 0 skipped)"
rule

SUITES_LOG="$LOG_DIR/suites.log"
(
    cd "$JAC" || exit 1
    "$PYTHON" -m jaclang test \
        tests/compiler/passes/test_graphmend_trap_integration.jac \
        tests/compiler/passes/test_graphmend_where_integration.jac \
        tests/compiler/passes/test_graphmend_defer_integration.jac \
        tests/compiler/passes/test_graphmend_import_integration.jac
) 2>&1 | tee "$SUITES_LOG"

# The native runner ends with "N passed, M skipped in T.TTs"; the pytest fence
# form is accepted too so the parse survives either runner.
SUMMARY="$(grep -E '^(=+ .*(passed|failed|error|skipped)|[0-9]+ (passed|failed|error|skipped|crashed).* in [0-9.]+s$)' "$SUITES_LOG" | tail -1)"
count() { printf '%s' "$SUMMARY" | grep -oE "[0-9]+ $1" | grep -oE '^[0-9]+' || true; }
PASSED="$(count passed)"; PASSED="${PASSED:-0}"
SKIPPED="$(count skipped)"; SKIPPED="${SKIPPED:-0}"

echo
if [ "$SKIPPED" -gt 0 ]; then
    fail "rule suites: $SKIPPED skipped (torch not visible to the toolchain)"
elif [ "$PASSED" -lt 18 ]; then
    fail "rule suites: $PASSED passed, expected 18 (summary: ${SUMMARY:-none found})"
else
    pass "rule suites: $PASSED passed, 0 skipped ([Trap] 6, [Where] 3, [Defer] 7, import 2)"
fi

if [ "$MODE" = "suites" ]; then
    rule
    echo "SUMMARY"
    rule
    printf '%s\n' "${RESULTS[@]}"
    exit "$FAILED"
fi

# ---------------------------------------------------------------------------
# Step 2: model rows
#
# Each row runs the forward pass through a counting backend in two isolated
# subprocesses, GraphMend off then on, and compares a SHA-256 fingerprint of
# the output tensor between the two.
#
# The expected values are the measured values recorded in artifact/README.md.
# Two of the five are deliberately NOT clean sweeps: longformer is 40% and clap
# is 0% in the paper's own Table 2, because what survives is the paper's
# declared out-of-scope category. A run that "fixed" them would be the surprise.
# ---------------------------------------------------------------------------
rule
echo "STEP 2  model rows (GraphMend off vs on, break counts and output hash)"
rule

if [ "$MODE" = "quick" ]; then
    EXPECT=(
        "t5-small 3 0"
        "Phi-4-mini-instruct 5 0"
    )
else
    EXPECT=(
        "t5-small 3 0"
        "biogpt 2 0"
        "Phi-4-mini-instruct 5 0"
        "longformer-base-4096 5 3"
        "clap-htsat-fused 2 2"
    )
fi

KEYS=()
for row in "${EXPECT[@]}"; do KEYS+=("$(printf '%s' "$row" | awk '{print $1}')"); done

echo "  models: ${KEYS[*]}"
echo "  the first run of each model is slow: GraphMend compiles the imported"
echo "  modeling code through the Jac front end on a cold cache. Progress is"
echo "  printed per model as it starts."
echo

EVAL_LOG="$LOG_DIR/run_eval.log"
(
    cd "$JAC" || exit 1
    "$PYTHON" -m paper_eval.run_eval "${KEYS[@]}"
) 2>&1 | tee "$EVAL_LOG"

echo
for row in "${EXPECT[@]}"; do
    key="$(printf '%s' "$row" | awk '{print $1}')"
    want_before="$(printf '%s' "$row" | awk '{print $2}')"
    want_after="$(printf '%s' "$row" | awk '{print $3}')"

    line="$(awk -v k="$key" '$1==k {print; exit}' "$EVAL_LOG")"
    if [ -z "$line" ]; then
        fail "$key: no result row (the model errored; see $EVAL_LOG)"
        continue
    fi
    got_before="$(printf '%s' "$line" | awk '{print $2}')"
    got_after="$(printf '%s' "$line" | awk '{print $3}')"
    got_ok="$(printf '%s' "$line" | awk '{print $5}')"

    if [ "$got_before" = "-" ]; then
        fail "$key: ERR row (the model failed to build or run; see $EVAL_LOG)"
    elif [ "$got_before" != "$want_before" ]; then
        fail "$key: $got_before breaks before, expected $want_before"
    elif [ "$got_after" != "$want_after" ]; then
        fail "$key: $got_before -> $got_after, expected $want_before -> $want_after"
    elif [ "$got_ok" != "yes" ]; then
        fail "$key: output fingerprint differs between the two arms"
    else
        pass "$key: $got_before -> $got_after breaks, output identical"
    fi
done

# A whole-run guard against the single most common way this measurement goes
# wrong silently: if graphmend_claim_imports never takes effect, nothing is
# transformed and EVERY row reads N -> N. That looks like a result, not a
# misconfiguration, so name it.
ALL_UNCHANGED=1
for row in "${EXPECT[@]}"; do
    key="$(printf '%s' "$row" | awk '{print $1}')"
    line="$(awk -v k="$key" '$1==k {print; exit}' "$EVAL_LOG")"
    b="$(printf '%s' "$line" | awk '{print $2}')"
    a="$(printf '%s' "$line" | awk '{print $3}')"
    [ -n "$line" ] && [ "$b" != "$a" ] && ALL_UNCHANGED=0
done
if [ "$ALL_UNCHANGED" = "1" ]; then
    echo
    echo "  Every row is unchanged. That is the signature of GraphMend not"
    echo "  reaching the model code at all. See gotcha 1 and gotcha 2 in"
    echo "  artifact/README.md before reading these numbers as a result."
fi

rule
echo "SUMMARY"
rule
printf '%s\n' "${RESULTS[@]}"
echo
if [ "$FAILED" = "0" ]; then
    echo "All checks passed. Full output kept in $LOG_DIR"
    echo "Next: the complete 21-row offline sweep, from the jac/ directory:"
    echo "  PYTHONPATH=\$PWD $PYTHON -m paper_eval.run_eval"
else
    echo "Some checks failed. Full output kept in $LOG_DIR"
    echo "artifact/README.md has a troubleshooting section; the two gotchas at"
    echo "the top of it account for most unexpected 0% rows."
fi
exit "$FAILED"

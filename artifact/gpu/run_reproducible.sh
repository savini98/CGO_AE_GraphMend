#!/usr/bin/env bash
#
# GraphMend CGO 2027 artifact: the GPU claims that reproduce.
#
#   bash artifact/gpu/run_reproducible.sh
#
# Needs one CUDA device. Every check below has a fixed expected value taken
# from artifact/RESULTS.md, and this script exits non-zero if any of them
# misses. It is the GPU counterpart of artifact/run_all.sh, which covers the
# CPU claims on the same terms.
#
# What this covers:
#
#   break elimination on GPU      t5-small 3 -> 0, MoLFormer-XL 5 -> 0,
#                                 Phi-4-mini 5 -> 0
#   CUDA-graph launches per fwd   4 -> 1, 50 -> 1, 5 -> 1
#   C8 cold start                 a speedup on all three models
#
# What it deliberately does NOT cover: C9 steady state and C10 throughput.
# Those do not reproduce the paper's Table 2, so gating on them would report a
# failure for something this artifact already documents as an open question.
# They are measured by run_open_questions.sh beside this file, which prints
# numbers and never gates.
#
# Cold start is measured with a PRIVATE TorchInductor cache per arm, which
# bench.py sets up. Sharing one cache between the arms inflates the ratio by
# roughly a factor of five, because the second arm reuses kernels the first
# compiled. See the cold-start section of artifact/RESULTS.md.
#
# These models are built from REAL pretrained weights, downloaded on first use.
# Phi-4-mini-instruct is about 7.7 GB; the other two are small. To run a subset
# on a machine where that download is unwelcome:
#
#   GM_GPU_MODELS="t5-small MoLFormer-XL-both10pct" bash artifact/gpu/run_reproducible.sh
#
# Every model in the list is still checked against its expected value, so a
# subset run is a narrower proof, not a weaker one.
#
# Environment overrides:
#   PYTHON=...          interpreter to use (default: python3)
#   GM_GPU_MODELS="..." space-separated model subset (default: all three)

set -uo pipefail

GPU_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$GPU_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
LOG_DIR="${TMPDIR:-/tmp}/graphmend-gpu-$$"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$REPO/jac${PYTHONPATH:+:$PYTHONPATH}"

read -r -a MODELS <<< "${GM_GPU_MODELS:-t5-small MoLFormer-XL-both10pct Phi-4-mini-instruct}"

FAILED=0
pass() { printf '  PASS  %s\n' "$1"; }
fail() { FAILED=1; printf '  FAIL  %s\n' "$1"; }
rule() { printf '\n%s\n' "------------------------------------------------------------------"; }

# ---------------------------------------------------------------------------
# Step 0: a real CUDA device
# ---------------------------------------------------------------------------
rule
echo "STEP 0  environment"
rule

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "  no such interpreter: $PYTHON"
    exit 2
fi

"$PYTHON" - <<'PY' || exit 2
import sys
try:
    import torch
except Exception as exc:
    sys.exit(f"  torch is not importable: {exc}")
if not torch.cuda.is_available():
    sys.exit(
        "  no CUDA device visible.\n"
        "  The break-elimination and correctness claims do not need a GPU;\n"
        "  use artifact/run_all.sh for those. If this is the CUDA container\n"
        "  without --gpus, see the device-passthrough recipe in the header of\n"
        "  artifact/Dockerfile.cuda."
    )
print(f"  torch       {torch.__version__} (cuda {torch.version.cuda})")
print(f"  device      {torch.cuda.get_device_name(0)}")
PY

echo "  models      ${MODELS[*]}"
echo "  logs        $LOG_DIR"

# ---------------------------------------------------------------------------
# Step 1: break elimination on the GPU
#
# Structural, so these are exact equalities rather than thresholds. This is the
# same measurement artifact/run_all.sh makes on CPU, repeated here because the
# latency claims are only meaningful if the transform actually fired on the
# device the timings come from.
# ---------------------------------------------------------------------------
rule
echo "STEP 1  graph breaks, GraphMend off vs on"
rule

COUNTS_JSON="$LOG_DIR/counts.json"
"$PYTHON" "$GPU_DIR/bench.py" --count --json "${MODELS[@]}" \
    > "$COUNTS_JSON" 2> "$LOG_DIR/counts.err"
tail -1 "$COUNTS_JSON" > "$COUNTS_JSON.line" 2>/dev/null

"$PYTHON" - "$COUNTS_JSON.line" <<'PY'
import json, sys
want = {
    "t5-small":                (3, 0),
    "MoLFormer-XL-both10pct":  (5, 0),
    "Phi-4-mini-instruct":     (5, 0),
}
try:
    data = json.load(open(sys.argv[1]))
except Exception as exc:
    print(f"  FAIL  could not parse bench output: {exc}")
    sys.exit(1)
if not data:
    print("  FAIL  bench produced no rows at all")
    sys.exit(1)
bad = 0
for key, r in data.items():
    if key not in want:
        print(f"  FAIL  {key}: no expected break count recorded for this model")
        bad = 1
        continue
    wb, wa = want[key]
    if r["off"].get("error") or r["on"].get("error"):
        print(f"  FAIL  {key}: {r['off'].get('error') or r['on'].get('error')}")
        bad = 1
        continue
    gb, ga = r["off"]["breaks"], r["on"]["breaks"]
    if (gb, ga) != (wb, wa):
        print(f"  FAIL  {key}: {gb} -> {ga} breaks, expected {wb} -> {wa}")
        bad = 1
    else:
        print(f"  PASS  {key}: {gb} -> {ga} breaks")
sys.exit(bad)
PY
[ $? -eq 0 ] || FAILED=1

# ---------------------------------------------------------------------------
# Step 2: cold start (C8) and CUDA-graph launches
#
# The launch count is the check that the transform reached the compiled
# program: identical counts across arms mean it did not, and any timing
# difference then is noise. It is exact.
#
# The cold-start ratio is a timing, so it is gated with a wide threshold rather
# than an expected value. Measured on an RTX 3090 the three models give 3.29x,
# 2.22x and 5.92x on the conservative compile-subtracted metric; the gate below
# is 1.5x, which fails a run where the transform did nothing while tolerating a
# slower or busier machine.
# ---------------------------------------------------------------------------
rule
echo "STEP 2  cold start and CUDA-graph launches"
rule
echo "  this compiles each model twice, once per arm, and is slow on a cold cache"
echo

TIME_JSON="$LOG_DIR/timing.json"
"$PYTHON" "$GPU_DIR/bench.py" --json "${MODELS[@]}" \
    > "$TIME_JSON" 2> "$LOG_DIR/timing.err"
tail -1 "$TIME_JSON" > "$TIME_JSON.line" 2>/dev/null

"$PYTHON" - "$TIME_JSON.line" <<'PY'
import json, sys
COLD_MIN = 1.5
want_launches = {
    "t5-small":               (4, 1),
    "MoLFormer-XL-both10pct": (50, 1),
    "Phi-4-mini-instruct":    (5, 1),
}
try:
    data = json.load(open(sys.argv[1]))
except Exception as exc:
    print(f"  FAIL  could not parse bench output: {exc}")
    sys.exit(1)
if not data:
    print("  FAIL  bench produced no rows at all")
    sys.exit(1)
bad = 0
for key, r in data.items():
    if key not in want_launches:
        print(f"  FAIL  {key}: no expected launch count recorded for this model")
        bad = 1
        continue
    if "cold_ms" not in r.get("off", {}) or "cold_ms" not in r.get("on", {}):
        print(f"  FAIL  {key}: no measurement "
              f"(off={r.get('off', {}).get('error')})")
        bad = 1
        continue
    off, on = r["off"], r["on"]

    lo, ln = off.get("cudagraph_launches"), on.get("cudagraph_launches")
    if lo is None or ln is None:
        print(f"  FAIL  {key}: no CUDA-graph launch count")
        bad = 1
    elif lo == ln:
        print(f"  FAIL  {key}: launches off={lo} on={ln}, identical means the "
              f"transform never reached the compiled program")
        bad = 1
    elif ln != 1:
        print(f"  FAIL  {key}: launches off={lo} on={ln}, expected on=1")
        bad = 1
    else:
        print(f"  PASS  {key}: launches {lo} -> {ln}")

    raw = off["cold_window_ms"] / on["cold_window_ms"]
    cons = off["cold_ms"] / on["cold_ms"]
    if cons < COLD_MIN:
        print(f"  FAIL  {key}: cold {cons:.2f}x conservative "
              f"({raw:.2f}x raw window), below the {COLD_MIN}x gate")
        bad = 1
    else:
        print(f"  PASS  {key}: cold {cons:.2f}x conservative, "
              f"{raw:.2f}x raw window")
sys.exit(bad)
PY
[ $? -eq 0 ] || FAILED=1

rule
echo "SUMMARY"
rule
if [ "$FAILED" = "0" ]; then
    echo "All GPU checks passed. Full output kept in $LOG_DIR"
    echo
    echo "C9 steady state and C10 throughput are NOT checked here, because they"
    echo "do not reproduce Table 2. To see those numbers:"
    echo "  bash artifact/gpu/run_open_questions.sh"
else
    echo "Some GPU checks failed. Full output kept in $LOG_DIR"
    echo "stderr is in $LOG_DIR/counts.err and $LOG_DIR/timing.err"
fi
exit "$FAILED"

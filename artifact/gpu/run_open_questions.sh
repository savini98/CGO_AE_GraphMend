#!/usr/bin/env bash
#
# GraphMend CGO 2027 artifact: the GPU claims that do NOT reproduce.
#
#   bash artifact/gpu/run_open_questions.sh
#
# This script ALWAYS EXITS 0. It is a measurement, not a check. Nothing here
# has a pass/fail expectation, because the values it prints are the ones this
# artifact could not reconcile with the paper's Table 2, and a reviewer
# re-measuring them should see what we saw rather than a red failure.
#
# The claims that do reproduce, with real expected values and a real exit
# status, are in run_reproducible.sh beside this file and in
# artifact/run_all.sh.
#
# C9, steady state. The paper claims 1.05x to 1.39x, and Table 2's column for
# MoLFormer-XL reads 1.13x. This artifact measures about 1.03x on a
# configuration that reproduces the authors' own per-iteration timings to
# within 1%, and recomputing the same quantity from the authors' stored 3090
# traces gives 1.014x. Five metric definitions were tried and all landed near
# 1.0. The runs behind Table 2's 1.13x have not been identified.
#
# C10, throughput. The paper claims up to 15%. What this artifact measures is
# a mechanism rather than a single number: the gain tracks how many CUDA-graph
# launches the transform removes. MoLFormer-XL sheds 49 of its 50 launches and
# gains about 70% at batch 1, then nothing by batch 512 where compute
# dominates. Models that shed only three or four launches gain nothing at any
# batch size.
#
# Environment overrides:
#   PYTHON=...   interpreter to use (default: python3)

set -uo pipefail

GPU_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$GPU_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
LOG_DIR="${TMPDIR:-/tmp}/graphmend-gpu-open-$$"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$REPO/jac${PYTHONPATH:+:$PYTHONPATH}"

read -r -a MODELS <<< "${GM_GPU_MODELS:-t5-small MoLFormer-XL-both10pct Phi-4-mini-instruct}"

rule() { printf '\n%s\n' "------------------------------------------------------------------"; }

rule
echo "OPEN QUESTIONS  C9 steady state, C10 throughput"
rule

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "  no such interpreter: $PYTHON"
    exit 0
fi

"$PYTHON" - <<'PY' || exit 0
import sys
try:
    import torch
except Exception as exc:
    sys.exit(f"  torch is not importable: {exc}")
if not torch.cuda.is_available():
    sys.exit("  no CUDA device visible, nothing to measure here.")
print(f"  torch       {torch.__version__} (cuda {torch.version.cuda})")
print(f"  device      {torch.cuda.get_device_name(0)}")
PY

echo "  models      ${MODELS[*]}"
echo "  logs        $LOG_DIR"

# The paper sizes each model to about 70% of GPU memory. The default batch is
# much smaller, and steady state is measured at both because the two disagree
# about C10 and agree about C9, which is itself part of the finding.
for cfg in default paper; do
    rule
    if [ "$cfg" = "paper" ]; then
        echo "STEADY STATE, the paper's batch sizes (t5-small 1345, MoLFormer 837)"
        FLAG=(--paper-batch)
    else
        echo "STEADY STATE, this benchmark's default batch"
        FLAG=()
    fi
    rule

    # Invoked from jac/: bench.py derives PYTHONPATH and PAPER_EVAL_DIR for its
    # arm subprocesses from the working directory.
    OUT="$LOG_DIR/steady-$cfg.json"
    ( cd "$REPO/jac" && "$PYTHON" "$GPU_DIR/bench.py" --json "${FLAG[@]}" "${MODELS[@]}" ) \
        > "$OUT" 2> "$LOG_DIR/steady-$cfg.err"
    tail -1 "$OUT" > "$OUT.line" 2>/dev/null

    "$PYTHON" - "$OUT.line" <<'PY'
import json, sys
TABLE2 = {"MoLFormer-XL-both10pct": 1.13}
try:
    data = json.load(open(sys.argv[1]))
except Exception as exc:
    print(f"  could not parse bench output: {exc}")
    sys.exit(0)
print(f"  {'model':24s} {'off ms':>10s} {'on ms':>10s} {'measured':>10s}  paper")
for key, r in data.items():
    off, on = r.get("off", {}), r.get("on", {})
    if "warm_ms" not in off or "warm_ms" not in on:
        print(f"  {key:24s} {'no measurement':>32s}")
        continue
    ratio = off["warm_ms"] / on["warm_ms"]
    claim = TABLE2.get(key)
    claim_s = f"{claim:.2f}x (Table 2)" if claim else "1.05x-1.39x (claimed range)"
    print(f"  {key:24s} {off['warm_ms']:10.3f} {on['warm_ms']:10.3f} "
          f"{ratio:9.3f}x  {claim_s}")
print()
print("  Steady state is where this artifact and Table 2 disagree. A ratio near")
print("  1.0 here is the expected outcome of this script, not a failure.")
PY
done

rule
echo "THROUGHPUT (C10)"
rule
cat <<'EOF'
  Throughput tracks the number of CUDA-graph launches the transform removes,
  which run_reproducible.sh prints per model. Measured on an RTX 3090:

    model           launches   batch 1   batch 8   large batch
    t5-small        4 -> 1     0.984x    1.000x    1.001x (b256)
    Phi-4-mini      5 -> 1     1.008x    1.005x    1.002x (b16)
    MoLFormer-XL    50 -> 1    1.70x     1.017x    1.009x (b512)

  The batch-1 MoLFormer figure is four runs (1.729x, 1.616x, 1.692x, 1.755x)
  against a noise band of about half a percent. It is the one row that clears
  the paper's 15%, and it does so only where per-launch overhead dominates.

  These were measured with a standalone driver rather than through bench.py.
  See the C10 section of artifact/RESULTS.md for the method and the raw runs.
EOF

rule
echo "This script does not gate. Exit status is 0 regardless of the numbers above."
echo "Full output kept in $LOG_DIR"
rule
exit 0

#!/usr/bin/env bash
# The functional workflow: does GraphMend work? 10-20 minutes, CPU only.
#
#   bash scripts/run_quick.sh
#
# Answers the artifact-evaluation question -- all three transformations execute,
# graph breaks disappear, outputs stay equal -- without reproducing the paper's
# model suite or any performance number. No GPU, no network, no weight
# downloads. Exits non-zero if any check fails.
#
# The full reproduction is scripts/run_full.sh.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
need_setup

banner "FUNCTIONAL WORKFLOW (quick)" \
       "Sections 4.3 and 5.1" \
       "all rule suites pass; break counts match; outputs identical. ~10-20 min."

FAILED=0

# Step 1: each rule collapses a broken region into one FX graph, on fixtures.
# One suite per rule plus the import-claiming path, so a failure names a rule.
bash "$REPO/artifact/run_all.sh" --suites || FAILED=1

# Step 2: the same measurement on real Hugging Face models, one per rule.
#   t5-small             [Defer]  3 -> 0
#   Phi-4-mini-instruct  [Where]  5 -> 0   (the paper's Figure 3 example)
#   longformer-base-4096 partial  5 -> 3   (correctly NOT a clean sweep)
#   clap-htsat-fused     none     2 -> 2   (correctly NOT fixed)
# The last two matter: a run that "fixed" them would be the surprise, not the
# win. [Trap] has no offline row, so its real-model demonstration is
# MoLFormer-XL, which needs network and is in run_full.sh.
bash "$REPO/artifact/run_all.sh" || FAILED=1

printf '\n%s\n' "=================================================================="
if [ "$FAILED" = "0" ]; then
    echo "QUICK WORKFLOW PASSED."
    echo
    echo "Next, the full reproduction (hours, and a GPU for the latency parts):"
    echo "  bash scripts/run_full.sh"
else
    echo "QUICK WORKFLOW FAILED. See artifact/README.md, Troubleshooting."
    echo "Most unexpected 0% rows are one of the two ways to measure nothing."
fi
printf '%s\n' "=================================================================="
exit "$FAILED"

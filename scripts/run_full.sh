#!/usr/bin/env bash
# The full reproduction workflow.
#
#   bash scripts/run_full.sh          # every claim the hardware allows
#
# Runs each per-claim script in turn and reports which ran. The CPU claims
# always run. The two GPU claims run only where a CUDA device is visible, and
# are reported as SKIPPED rather than failed otherwise, since the paper's
# hardware is not a precondition for the artifact's primary claims.
#
# Hours, not minutes. By default the rows needing network access and
# trust_remote_code are skipped; GM_NETWORK=1 includes them.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
need_setup

RESULTS=()
run_step() {
    local label="$1"; shift
    if "$@"; then RESULTS+=("PASS     $label"); else RESULTS+=("FAIL     $label"); fi
}

# Claim 1 is one script. Break elimination, output correctness and full-graph
# capture come from the same two-arm run over the same models, so splitting
# them into separate entry points would compile every model three times to
# report three columns of one result.
# Network rows are opt-in: without GM_NETWORK the run stays offline.
CLAIM1_ARGS=()
[ -n "${GM_NETWORK:-}" ] || CLAIM1_ARGS+=(--offline)
run_step "Claim 1: breaks eliminated, output identical, full graph (5.1, 5.6)" \
    bash "$REPO/artifact/run_break_analysis.sh" "${CLAIM1_ARGS[@]}"

run_step "compiler overhead (Figure 10, Section 5.7)" \
    bash "$REPO/scripts/run_compiler_overhead.sh"

if "$PYTHON" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    run_step "latency (Table 2, Sections 5.2-5.3)" bash "$REPO/scripts/run_latency.sh"
    run_step "throughput (Figure 9, Section 5.4)"  bash "$REPO/scripts/run_throughput.sh"
else
    RESULTS+=("SKIPPED  latency (Table 2) -- no CUDA device")
    RESULTS+=("SKIPPED  throughput (Figure 9) -- no CUDA device")
fi

printf '\n%s\n' "=================================================================="
echo "FULL WORKFLOW SUMMARY"
printf '%s\n' "=================================================================="
printf '%s\n' "${RESULTS[@]}"
case " ${RESULTS[*]} " in *"FAIL "*) exit 1 ;; esac
exit 0

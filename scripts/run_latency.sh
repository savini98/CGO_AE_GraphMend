#!/usr/bin/env bash
# Cold-start and steady-state forward-pass latency: paper Table 2, Sections 5.2-5.3.
#
#   bash scripts/run_latency.sh
#   GM_GPU_MODELS="t5-small" bash scripts/run_latency.sh    # a subset
#
# NEEDS AN NVIDIA GPU. Builds each model from real pretrained weights and gives
# each arm a private TorchInductor cache, so the second arm's cold start is
# genuinely cold rather than a cache hit.
#
# Magnitudes depend on the card, the driver and the batch size, so this gates on
# the hardware-independent part -- breaks reaching zero, and CUDA-graph launches
# per forward collapsing to one -- and reports the timings without gating them.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
need_setup
need_cuda

banner "LATENCY (cold start and steady state)" \
       "Table 2, Cold Start and Steady State columns; Sections 5.2 and 5.3" \
       "breaks -> 0 and launches -> 1 exactly; cold-start speedup well above 1."

exec bash "$REPO/artifact/gpu/run_reproducible.sh"

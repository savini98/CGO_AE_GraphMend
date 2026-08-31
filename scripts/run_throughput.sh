#!/usr/bin/env bash
# End-to-end inference throughput: paper Figure 9, Section 5.4.
#
#   bash scripts/run_throughput.sh
#   bash scripts/run_throughput.sh t5-small MoLFormer-XL-both10pct
#
# NEEDS AN NVIDIA GPU. Generative models produce 100 output tokens with greedy
# decoding, as in the paper; encoders are measured in samples per second.
#
# Figure 9 is 24 models across three GPUs. This measures the models you name on
# the card you have, so it reproduces the distribution's shape rather than its
# individual bars: the gain tracks how many CUDA-graph launches the transform
# removes, and shrinks as the batch grows (Section 5.4, Amdahl).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
need_setup
need_cuda

banner "THROUGHPUT" \
       "Figure 9; Section 5.4" \
       "a gain that tracks the number of CUDA-graph launches eliminated."

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(t5-small MoLFormer-XL-both10pct Phi-4-mini-instruct)

cd "$REPO" || exit 1
exec "$PYTHON" artifact/gpu/bench.py --throughput "${MODELS[@]}"

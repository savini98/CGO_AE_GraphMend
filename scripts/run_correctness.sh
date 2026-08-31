#!/usr/bin/env bash
# Output equality between the two arms: paper Section 5.1.
#
#   bash scripts/run_correctness.sh [model_key ...]
#
# Same two-arm run as run_break_analysis.sh: it compares a SHA-256 fingerprint
# of the output between arms, with both arms pinned to the same weights and the
# same inputs. Generative rows compare a greedy-decoded token sequence over 16
# steps rather than a single forward, so the `compared` column reads `tokens`
# for those and `logits` otherwise. Read the `output_ok` column.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
need_setup

banner "OUTPUT CORRECTNESS" \
       "Section 5.1, output correctness validation" \
       "output_ok = yes on EVERY row, including rows that fix nothing."

cd "$REPO" || exit 1
exec "$PYTHON" -m paper_eval.run_eval "$@"

#!/usr/bin/env bash
# Graph-break elimination: paper Table 2.
#
#   bash scripts/run_break_analysis.sh                 # 21 offline rows
#   bash scripts/run_break_analysis.sh t5-small biogpt # named rows
#   GM_NETWORK=1 bash scripts/run_break_analysis.sh    # add the 6 network rows
#
# Each model's forward pass runs through a counting TorchDynamo backend in two
# isolated subprocesses, GraphMend off then on. breaks = FX graphs - 1.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
need_setup

banner "GRAPH-BREAK ELIMINATION" \
       "Table 2, Fixed(%) column; Section 5.1" \
       "21 models fully fixed, 3 partial, 3 unfixed. Offline total 89 -> 19 (78%)."

cd "$REPO" || exit 1
if [ -n "${GM_NETWORK:-}" ] && [ "$#" -eq 0 ]; then
    set -- Florence-2 MoLFormer-XL-both10pct chronos-bolt-small \
           Qwen-Audio-Chat stella-en-400M-v5 moe-minicpm-x4-base
fi
exec "$PYTHON" -m paper_eval.run_eval "$@"

#!/usr/bin/env bash
# Full-graph capture for serving frameworks: paper Section 5.6.
#
#   bash scripts/run_fullgraph.sh [model_key ...]
#
# vLLM and SGLang require torch.compile(fullgraph=True), which a single graph
# break defeats. Attempts it per arm with backend="eager", which isolates
# Dynamo's capture from backend compilation and so needs no GPU and is
# deterministic.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
need_setup

banner "FULL-GRAPH CAPTURE" \
       "Section 5.6" \
       "off FAILS and on SUCCEEDS on every row. Both-pass or both-fail is a FAIL."

cd "$REPO" || exit 1
exec "$PYTHON" -m paper_eval.run_fullgraph "$@"

#!/usr/bin/env bash
# GraphMend's own compilation overhead: paper Figure 10, Section 5.7.
#
#   bash scripts/run_compiler_overhead.sh [model_key ...]
#
# Times the same entry program end to end under standard Python and under
# `jac run`, cold (empty compiler cache) and cached (artifacts reused). Each
# cold run gets a private XDG_CACHE_HOME, so this neither reads nor destroys
# the reviewer's own cache.
#
# CPU only. No GPU, no weights, no network for the default model set.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
need_setup

banner "COMPILER OVERHEAD" \
       "Figure 10; Section 5.7" \
       "mean ~11.5% cold and ~1.1% cached, over standard Python."

cd "$REPO" || exit 1
exec "$PYTHON" -m paper_eval.run_overhead "$@"

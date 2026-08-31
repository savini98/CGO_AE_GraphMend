#!/usr/bin/env bash
#
# GraphMend CGO 2027 artifact: prepare the toolchain.
#
#   bash scripts/setup.sh
#
# The toolchain is not vendored here. This artifact ships two things instead:
#
#   jaseci/                   the upstream Jac compiler, a git submodule frozen
#                             at the commit GraphMend was developed against
#   patches/graphmend.patch   everything GraphMend adds to it
#
# That split is deliberate. The patch IS the contribution: 203 files, of which
# 166 are new, and the 37 modified files are the integration points in the
# compiler driver, the pass schedule, the Python front end and the runtime.
# Reading it tells you exactly what this paper changes, which a vendored copy
# of a 4,700-file compiler does not.
#
# This script is idempotent: it detects an already-patched tree and does
# nothing, so it is safe to re-run.
#
# Environment overrides:
#   PYTHON=...   interpreter used for the typeshed fetch (default: python3)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
SUB="$REPO/jaseci"
PATCH="$REPO/patches/graphmend.patch"
# The commit the patch was generated against. Applying it to any other commit
# is the one way this setup fails confusingly, so it is checked rather than
# assumed.
PINNED="e2b6b9f4bdec510622410f046c8bd5427980c33f"
# A file that exists only after the patch is applied. Cheaper and more honest
# than a stamp file, which can outlive the tree it describes.
SENTINEL="$SUB/jac/jaclang/compiler/passes/graphmend/scope_facts.jac"

step() { printf '\n==> %s\n' "$1"; }

step "submodule"
if [ ! -e "$SUB/jac/README.md" ]; then
    echo "    fetching jaseci at $PINNED"
    git -C "$REPO" submodule update --init --depth 1 jaseci 2>/dev/null \
        || git -C "$REPO" submodule update --init jaseci
else
    echo "    already checked out"
fi

# Verify the pin where git can see it. Inside a Docker build context it cannot:
# .dockerignore drops .git, so the copied submodule is a plain directory. The
# patch application below is the real check in that case -- it is generated
# against $PINNED and will not apply to a different tree -- so a missing repo
# is reported, not treated as a failure.
if HAVE="$(git -C "$SUB" rev-parse HEAD 2>/dev/null)"; then
    if [ "$HAVE" != "$PINNED" ]; then
        echo "    ERROR: jaseci is at $HAVE, expected $PINNED"
        echo "    patches/graphmend.patch is generated against the pinned commit"
        echo "    and will not apply cleanly to another one. Fix with:"
        echo "      git -C jaseci fetch origin $PINNED && git -C jaseci checkout $PINNED"
        exit 1
    fi
    echo "    at $PINNED (verified)"
else
    echo "    at $PINNED (assumed: not a git checkout here, e.g. a Docker"
    echo "    build context or an unpacked archive; the patch below is the check)"
fi

step "graphmend patch"
if [ -f "$SENTINEL" ]; then
    echo "    already applied (found ${SENTINEL#"$REPO"/})"
elif ( cd "$SUB" && git apply --check -p1 "$PATCH" ) 2>/dev/null; then
    # `git apply` works outside a repository, which is what makes one script
    # serve both a clone and a Docker build context.
    ( cd "$SUB" && git apply -p1 "$PATCH" )
    echo "    applied: $(grep -c '^diff --git' "$PATCH") files"
else
    echo "    ERROR: patches/graphmend.patch does not apply to $SUB."
    echo
    echo "    Most likely the submodule is not at $PINNED,"
    echo "    or its working tree has been modified. In a git checkout:"
    echo "      git -C jaseci checkout -- . && git -C jaseci clean -fd"
    echo "      git -C jaseci checkout $PINNED"
    echo "    then re-run this script."
    exit 1
fi

# The submodule's working tree is now intentionally dirty: the patch is applied
# in place rather than committed, so `git status` in jaseci/ shows the diff and
# `git -C jaseci diff` is another way to read the contribution.

step "typeshed stubs"
if [ -d "$SUB/jac/jaclang/vendor/typeshed/stdlib" ]; then
    echo "    already present"
else
    "$PYTHON" "$REPO/artifact/fetch_typeshed.py" "$SUB/jac"
fi

step "ready"
cat <<EOF
    toolchain   $SUB/jac
    harness     $REPO/paper_eval

    Next:
      bash artifact/run_all.sh --suites    # fastest real check
      bash artifact/run_all.sh             # 4 rule suites + 5 models

    torch and transformers must be importable. If they are not, use the
    container instead, which pins them:
      docker build -f artifact/Dockerfile.cpu -t graphmend-cpu .
EOF

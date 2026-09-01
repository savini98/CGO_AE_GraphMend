#!/usr/bin/env bash
# Build the archival tarball for a Zenodo deposit.
#
#   bash scripts/make_archive.sh [output.tar.gz]
#
# `git archive` does NOT include submodule contents, so an archive made with it
# would ship an empty jaseci/ and be unusable. This flattens the submodule into
# the tarball, so the deposit stands alone and its DOI does not depend on the
# submodule remaining reachable -- which is the point of an archival copy, and
# what the ACM Artifacts Available badge asks for that a repository alone does
# not provide.
#
# The pinned commit is recorded in ARCHIVE_INFO inside the tarball, so the
# flattened copy still says exactly what it was built from.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$REPO/graphmend-cgo2027-artifact.tar.gz}"
STAGE="$(mktemp -d)"
NAME="graphmend-cgo2027-artifact"
trap 'rm -rf "$STAGE"' EXIT

echo "==> staging tracked files"
git -C "$REPO" archive --format=tar --prefix="$NAME/" HEAD | tar -x -C "$STAGE"

echo "==> flattening the jaseci submodule"
SUB_SHA="$(git -C "$REPO/jaseci" rev-parse HEAD)"
rm -rf "${STAGE:?}/$NAME/jaseci"
git -C "$REPO/jaseci" archive --format=tar --prefix="$NAME/jaseci/" HEAD | tar -x -C "$STAGE"

# The typeshed stdlib stubs are gitignored in BOTH repositories, so neither
# `git archive` picks them up -- and without them no Jac compilation works at
# all. A deposit that still needs a network fetch to run is not an archival
# copy, so materialize them into the staged tree here.
echo "==> materializing the typeshed stubs"
if [ -d "$REPO/jaseci/jac/jaclang/vendor/typeshed/stdlib" ]; then
    cp -R "$REPO/jaseci/jac/jaclang/vendor/typeshed/stdlib" \
          "$STAGE/$NAME/jaseci/jac/jaclang/vendor/typeshed/stdlib"
else
    python3 "$REPO/artifact/fetch_typeshed.py" "$STAGE/$NAME/jaseci/jac"
fi
TYPESHED_N="$(find "$STAGE/$NAME/jaseci/jac/jaclang/vendor/typeshed/stdlib" -type f | wc -l | tr -d ' ')"
echo "    $TYPESHED_N stub files"

cat > "$STAGE/$NAME/ARCHIVE_INFO" <<INFO
GraphMend, CGO 2027 artifact -- archival copy

artifact commit   $(git -C "$REPO" rev-parse HEAD)
jaseci submodule  $SUB_SHA  (upstream jaseci-labs/jaseci, unpatched)
typeshed stubs    $TYPESHED_N files, bundled
built             $(date -u +%Y-%m-%dT%H:%M:%SZ)

The jaseci/ tree here is the UNPATCHED upstream toolchain at the pinned commit.
patches/graphmend.patch is GraphMend. Run scripts/setup.sh to apply it; it
skips the submodule fetch because the tree is already present, and detects an
already-patched tree, so it is safe to re-run.

This copy is self-contained: the typeshed stubs are bundled, so setup.sh needs
no network. It does need `git`, which is how the patch is applied.
INFO

echo "==> writing $OUT"
# COPYFILE_DISABLE and --no-xattrs: macOS tar otherwise writes an AppleDouble
# `._name` companion for every file carrying an extended attribute, and a
# checkout on this machine carries com.apple.provenance on all of them. Those
# companions are binary, they extract as real files next to the sources, and
# the Jac compiler globs `*.impl.jac` -- so it would pick up `._foo.impl.jac`
# and fail with UnicodeDecodeError on a tree that looks correct.
COPYFILE_DISABLE=1 tar --no-xattrs -czf "$OUT" -C "$STAGE" "$NAME" 2>/dev/null \
    || COPYFILE_DISABLE=1 tar -czf "$OUT" -C "$STAGE" "$NAME"

# Fail loudly rather than shipping a broken deposit.
if tar -tzf "$OUT" | grep -q '/\._'; then
    echo "    ERROR: the archive contains AppleDouble ._* files, which break"
    echo "    the Jac compiler on extraction. Do not upload this."
    exit 1
fi
echo "    $(du -h "$OUT" | cut -f1)  $OUT"
echo
echo "Upload this to Zenodo, then fill {{DOI}} into the paper's artifact appendix."

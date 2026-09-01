"""Materialize the vendored typeshed stdlib stubs into a source checkout.

WHY THIS EXISTS
---------------
`jaclang/vendor/typeshed/stdlib/` is gitignored. Only `PIN`, `TARBALL_SHA256`,
`LICENSE` and `PROVENANCE.md` are tracked; the stubs themselves are fetched at
the pinned commit. Type inference is on the critical path of every Jac
compilation, so without them the toolchain gets through its bootstrap schedule
and then dies on the first real compile with:

    TypeshedUnavailableError: type inference is on the critical path of every
    compilation, but the vendored typeshed stdlib stubs are missing ...

which means `jac run` and `jac test` do not work, which means no model row and
no rule suite can be measured. A fresh `git clone` of this repository hits that
wall, and so did the first build of artifact/Dockerfile.

The repository's own answer is `zig build fetch-typeshed`, which drives the
payload tool at jaclang/dist/payload/. That tool is itself written in Jac, and
pulling a Zig toolchain plus a python-build-standalone tree into a container
just to obtain a directory of .pyi files is a large amount of machinery for the
job. This script does exactly what `fetch_typeshed` in
jaclang/dist/payload/impl/fetch.impl.jac does, in the standard library only, so
it runs anywhere a bare Python does.

It is deliberately a re-implementation of a pinned, checksummed operation and
not a second source of truth: PIN and TARBALL_SHA256 are read from the vendor
directory, never hardcoded here. If the repository bumps the pin, this script
follows it with no edit.

INTEGRITY
---------
The checksum is over the DECOMPRESSED tar, not the gzip envelope, matching
TARBALL_SHA256 and the payload tool. Git's archive output is content-stable for
a commit, so this pins the exact tree. The hash is verified BEFORE anything is
extracted, so a swapped tarball never reaches the filesystem.

USAGE
-----
    python artifact/fetch_typeshed.py <repo-root>

<repo-root> is the directory holding `jaclang/`, i.e. the `jac/` directory of
this repository. Idempotent: it returns immediately if the stubs are already
present at the pinned commit.
"""

import gzip
import hashlib
import io
import os
import shutil
import sys
import tarfile
import urllib.request

TARBALL_BASE = "https://codeload.github.com/python/typeshed/tar.gz"
VENDOR_REL = os.path.join("jaclang", "vendor", "typeshed")
USER_AGENT = "jac-payload-tool"


def die(msg):
    sys.stderr.write("fetch-typeshed: " + msg + "\n")
    raise SystemExit(1)


def read_trimmed(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return None


def main(argv):
    if len(argv) != 2:
        die("usage: fetch_typeshed.py <repo-root>   (the directory holding jaclang/)")
    root = argv[1]
    vendor = os.path.join(root, VENDOR_REL)

    commit = read_trimmed(os.path.join(vendor, "PIN"))
    if not commit:
        die("no PIN at %s/PIN (is <repo-root> the jac/ directory?)" % vendor)
    expected = read_trimmed(os.path.join(vendor, "TARBALL_SHA256"))
    if not expected:
        die("no TARBALL_SHA256 at %s/TARBALL_SHA256" % vendor)

    stdlib_dst = os.path.join(vendor, "stdlib")
    # The same two-part up-to-date check the payload tool uses: the stamp alone
    # would call a half-extracted tree current, and VERSIONS alone would not
    # notice a pin bump.
    #
    # os.path.isfile follows symlinks, which is what we want. A developer
    # checkout may have `stdlib` symlinked at a shared tree to avoid fetching it
    # per git worktree, and if that link resolves it is a perfectly good set of
    # stubs -- the fetch is skipped and nothing is downloaded. The case this
    # does NOT accept is the same symlink copied somewhere its target does not
    # exist, which is exactly what `COPY jaseci /opt/artifact/jaseci` produces from such
    # a checkout: an absolute link into the developer's home directory, dangling
    # inside the image. That resolves to nothing, so the fetch proceeds and
    # replaces it below.
    stamp = os.path.join(stdlib_dst, ".typeshed-sha")
    if os.path.isfile(os.path.join(stdlib_dst, "VERSIONS")) and read_trimmed(stamp) == commit:
        print("fetch-typeshed: already at %s" % commit)
        return 0

    url = "%s/%s" % (TARBALL_BASE, commit)
    print("fetch-typeshed: fetching typeshed @ %s" % commit)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req) as resp:
            gz = resp.read()
    except Exception as exc:
        die("download failed: %s: %s" % (type(exc).__name__, exc))

    try:
        tar_bytes = gzip.decompress(gz)
    except Exception as exc:
        die("gzip decompress failed: %s: %s" % (type(exc).__name__, exc))

    actual = hashlib.sha256(tar_bytes).hexdigest()
    if actual != expected:
        die(
            "tarball checksum mismatch @ %s\n  expected %s\n  actual   %s"
            % (commit, expected, actual)
        )

    # `@tests` directories are stub test fixtures, not stubs. The payload tool
    # drops them (skip_typeshed_tests in jaclang/dist/payload/impl/fsutil.impl.jac)
    # and so does this, so the two produce the same tree.
    prefix = "stdlib/"
    staged = os.path.join(vendor, ".ts-extract")
    shutil.rmtree(staged, ignore_errors=True)
    os.makedirs(staged, exist_ok=True)
    written = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                # Strip the single leading component the GitHub tarball adds
                # ("typeshed-<commit>/"), matching extract_tar(..., 1, False).
                parts = member.name.split("/", 1)
                if len(parts) != 2:
                    continue
                rel = parts[1]
                if not rel.startswith(prefix) or "@tests" in rel:
                    continue
                rel = rel[len(prefix):]
                # Refuse anything that would escape the destination. The tar is
                # checksum-pinned, so this is belt-and-braces, but extracting an
                # archive without the check is the kind of thing that is only
                # safe until the pin is bumped by someone in a hurry.
                dst = os.path.normpath(os.path.join(staged, rel))
                if not dst.startswith(os.path.abspath(staged) + os.sep) and dst != staged:
                    dst_abs = os.path.abspath(dst)
                    if not dst_abs.startswith(os.path.abspath(staged) + os.sep):
                        die("refusing path outside the destination: %s" % member.name)
                src = tf.extractfile(member)
                if src is None:
                    continue
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, "wb") as out:
                    shutil.copyfileobj(src, out)
                written += 1

        if not os.path.isfile(os.path.join(staged, "VERSIONS")):
            die("tarball has no stdlib/VERSIONS (bad commit?)")

        # Swap in only after a complete, verified extraction, so an interrupted
        # run leaves the previous tree rather than a partial one.
        #
        # The symlink case has to be unlinked rather than rmtree'd: rmtree on a
        # symlink raises NotADirectoryError, which ignore_errors would swallow,
        # leaving the link in place for os.rename to fail on. islink is checked
        # first because it is the only one of the two that is true for a link to
        # a directory.
        if os.path.islink(stdlib_dst):
            os.unlink(stdlib_dst)
        else:
            shutil.rmtree(stdlib_dst, ignore_errors=True)
        os.makedirs(os.path.dirname(stdlib_dst), exist_ok=True)
        os.rename(staged, stdlib_dst)
    finally:
        shutil.rmtree(staged, ignore_errors=True)

    with open(os.path.join(stdlib_dst, ".typeshed-sha"), "w", encoding="utf-8") as fh:
        fh.write(commit)
    print("fetch-typeshed: ready (%s, %d files)" % (commit, written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

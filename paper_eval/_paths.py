"""Locations of the toolchain and the harness.

Each arm runs as a subprocess under `jac run` and needs two directories on
PYTHONPATH: the patched jaclang toolchain, and the artifact root so that
`paper_eval` imports inside the arm. Both are derived from this file's own
location, so the harness runs from any working directory.

GM_JACLANG_DIR overrides the toolchain location.
"""

import os

#: The artifact repository root: the directory containing `paper_eval/`.
ARTIFACT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The directory containing `jaclang/`, i.e. what goes on PYTHONPATH so that
#: `python -m jaclang` is the patched toolchain and not a pip-installed one.
#: Every released jaclang on PyPI predates GraphMend, so this must win.
JACLANG_DIR = os.environ.get(
    "GM_JACLANG_DIR", os.path.join(ARTIFACT_ROOT, "jaseci", "jac")
)

#: PYTHONPATH for an arm subprocess: toolchain first, then the artifact root.
ARM_PYTHONPATH = os.pathsep.join([JACLANG_DIR, ARTIFACT_ROOT])


def check() -> None:
    """Fail early and legibly if setup.sh has not been run.

    Without this the first symptom is `No module named jaclang` from inside an
    arm subprocess, which reads as a broken harness rather than as a missing
    setup step.
    """
    if not os.path.isdir(os.path.join(JACLANG_DIR, "jaclang")):
        raise SystemExit(
            f"no jaclang toolchain at {JACLANG_DIR}\n"
            "Run `bash scripts/setup.sh` first: it checks out the pinned jaseci\n"
            "submodule, applies patches/graphmend.patch, and fetches the typeshed\n"
            "stubs. Set GM_JACLANG_DIR to use a tree kept elsewhere."
        )
    gm = os.path.join(JACLANG_DIR, "jaclang", "compiler", "passes", "graphmend")
    if not os.path.isdir(gm):
        raise SystemExit(
            f"jaclang at {JACLANG_DIR} has no GraphMend passes.\n"
            "The submodule is checked out but patches/graphmend.patch has not\n"
            "been applied. Run `bash scripts/setup.sh`."
        )

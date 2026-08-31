"""Run why.py for one model under `jac run`, in the requested GraphMend mode.

    python -m paper_eval.run_why <model_key> [on|off]

Same reason run_eval.py exists: the mode has to be carried by a jac.toml next
to a Jac-compiled entry program, because there is no CLI switch for GraphMend
and running the entry under plain CPython leaves every [Defer] rewrite inert.
"""
import os
import shutil
import subprocess
import sys
import tempfile

from paper_eval import _paths

# Written per arm into a private temp dir. The [dev] stanza pins the compiler
# to the patched toolchain by absolute path: config resolution takes the
# NEAREST jac.toml, so without it this file would shadow the repository's and
# the arm could fall back to a linked dev binary's own checkout -- silently, and
# with a jaclang that predates GraphMend.
_JAC_TOML = """\
[dev]
jaclang_source = "{src}"

[run]
graphmend = {on}
graphmend_claim_imports = {on}
"""


def main(key: str, mode: str = "off") -> int:
    workdir = tempfile.mkdtemp(prefix=f"gmwhy_{key.replace('/', '_')}_{mode}_")
    try:
        with open(os.path.join(workdir, "jac.toml"), "w") as fh:
            fh.write(_JAC_TOML.format(on="true" if mode == "on" else "false",
                                      src=_paths.JACLANG_DIR))
        shutil.copy(os.path.join(_paths.ARTIFACT_ROOT, "paper_eval", "why.py"),
                    os.path.join(workdir, "why.py"))
        env = dict(os.environ, PYTHONPATH=_paths.ARM_PYTHONPATH,
                   PAPER_EVAL_DIR=_paths.ARTIFACT_ROOT,
                   GM_MODEL=key, GM_MODE=mode)
        p = subprocess.run([sys.executable, "-m", "jaclang", "run", "why.py"],
                           text=True, env=env, cwd=workdir)
        return p.returncode
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    _paths.check()
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "off"))

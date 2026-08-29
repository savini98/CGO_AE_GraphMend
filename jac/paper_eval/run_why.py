"""Run why.py for one model under `jac run`, in the requested GraphMend mode.

    PYTHONPATH=$PWD python -m paper_eval.run_why <model_key> [on|off]

Same reason run_eval.py exists: the mode has to be carried by a jac.toml next
to a Jac-compiled entry program, because there is no CLI switch for GraphMend
and running the entry under plain CPython leaves every [Defer] rewrite inert.
"""
import os
import shutil
import subprocess
import sys
import tempfile

_JAC_TOML = """\
[run]
graphmend = {on}
graphmend_claim_imports = {on}
"""


def main(key: str, mode: str = "off") -> int:
    repo = os.getcwd()
    workdir = tempfile.mkdtemp(prefix=f"gmwhy_{key.replace('/', '_')}_{mode}_")
    try:
        with open(os.path.join(workdir, "jac.toml"), "w") as fh:
            fh.write(_JAC_TOML.format(on="true" if mode == "on" else "false"))
        shutil.copy(os.path.join(repo, "paper_eval", "why.py"),
                    os.path.join(workdir, "why.py"))
        env = dict(os.environ, PYTHONPATH=repo, PAPER_EVAL_DIR=repo,
                   GM_MODEL=key, GM_MODE=mode)
        p = subprocess.run([sys.executable, "-m", "jaclang", "run", "why.py"],
                           text=True, env=env, cwd=workdir)
        return p.returncode
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "off"))

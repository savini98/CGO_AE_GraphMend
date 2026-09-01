"""C11, full-graph capture for serving frameworks (paper 5.6).

    python -m paper_eval.run_fullgraph            # all rows
    python -m paper_eval.run_fullgraph t5-small   # some rows

vLLM requires model code to be capturable via `torch.compile(fullgraph=True)`
to use its compilation and CUDA-Graph replay path, and SGLang's piecewise CUDA
graph relies on the same capture. A single graph break fails that requirement,
so "did the breaks go away" and "can this model be served" are the same
question asked twice.

Each row runs the model twice, GraphMend off then on, under `jac run`, exactly
as `run_eval` does. The expected result is asymmetric and that is the point:

    off  ->  FAILS to capture a full graph
    on   ->  captures a full graph

A row where both arms succeed means the model had no breaks to begin with, and
a row where both fail means GraphMend did not remove them all. Either way the
row fails, so this cannot pass by accident.

`backend="eager"` isolates Dynamo's graph capture from backend compilation, so
no GPU is needed and the result is deterministic: it is a property of the
source, not of the hardware.

Exit status is 0 only if every row shows off FAIL and on OK.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

from paper_eval import _paths
from paper_eval.registry import MODELS

# Written per arm into a private temp dir. The [dev] stanza pins the compiler
# to the patched toolchain by absolute path: config resolution takes the
# NEAREST jac.toml, so without it this file would shadow the repository's and
# the arm could fall back to a linked dev binary's own checkout -- silently, and
# with a jaclang that predates GraphMend.
_JAC_TOML = """[dev]
jaclang_source = "{src}"

[run]
graphmend = true
graphmend_claim_imports = {on}
"""

# The 21 rows the paper reports as fully fixed, minus those whose breaks sit on
# the generation path (entry.py reports those as skipped, see its GM_CHECK
# block). Rows that are only partially fixed are excluded on purpose: they
# still contain breaks after the transform, so full-graph capture must fail for
# them and that is not evidence about this claim.
PARTIAL = {"longformer-base-4096", "grounding-dino", "grounding-dino-base",
           "clap-htsat-fused", "moe-minicpm-x4-base", "stella-en-400M-v5"}


def _run(key, mode):
    workdir = tempfile.mkdtemp(prefix=f"gmfg_{key.replace('/', '_')}_{mode}_")
    try:
        with open(os.path.join(workdir, "jac.toml"), "w") as fh:
            fh.write(_JAC_TOML.format(on="true" if mode == "on" else "false",
                                      src=_paths.JACLANG_DIR))
        shutil.copy(os.path.join(_paths.ARTIFACT_ROOT, "paper_eval", "entry.py"),
                    os.path.join(workdir, "entry.py"))
        env = dict(os.environ, PYTHONPATH=_paths.ARM_PYTHONPATH,
                   PAPER_EVAL_DIR=_paths.ARTIFACT_ROOT,
                   GM_MODEL=key, GM_CHECK="fullgraph")
        p = subprocess.run([sys.executable, "-m", "jaclang", "run", "entry.py"],
                           capture_output=True, text=True, env=env, cwd=workdir)
        for line in reversed(p.stdout.strip().splitlines()):
            if line.startswith("GMRESULT "):
                return json.loads(line[len("GMRESULT "):])
        tail = (p.stderr.strip() or p.stdout.strip())[-200:]
        return {"key": key, "fullgraph": "error",
                "why": tail or f"no GMRESULT (exit {p.returncode})"}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# Rows that download weights or remote code. Named here rather than in the
# caller so the break-elimination step and this one cannot disagree about the
# set: both take --offline to mean the same thing.
NETWORK = {"MoLFormer-XL-both10pct", "Florence-2", "Qwen-Audio-Chat",
           "chronos-bolt-small", "moe-minicpm-x4-base", "stella-en-400M-v5"}


def main(argv):
    offline = "--offline" in argv
    argv = [a for a in argv if a != "--offline"]
    bad = [a for a in argv if a.startswith("-")]
    if bad:
        sys.exit(f"unknown option(s): {', '.join(bad)}")
    keys = argv or [k for k in MODELS if k not in PARTIAL]
    if offline:
        keys = [k for k in keys if k not in NETWORK]
    if not keys:
        sys.exit("no rows selected (every named row needs network, "
                 "and --offline was given)")
    print(f"{'model':32s} {'off':>10s} {'on':>10s}  verdict")
    print("-" * 68)
    failed = 0
    for key in keys:
        off = _run(key, "off")
        on = _run(key, "on")
        o, n = off.get("fullgraph"), on.get("fullgraph")
        if o == "skipped" or n == "skipped":
            print(f"{key:32s} {'-':>10s} {'-':>10s}  SKIP  "
                  f"({off.get('why') or on.get('why')})")
            continue
        if o == "failed" and n == "ok":
            verdict = "PASS"
        else:
            verdict = "FAIL"
            failed += 1
        print(f"{key:32s} {str(o):>10s} {str(n):>10s}  {verdict}")
        if verdict == "FAIL":
            for label, res in (("off", off), ("on", on)):
                if res.get("why"):
                    print(f"    {label}: {res['why'][:150]}")
    print()
    if failed:
        print(f"{failed} row(s) did not show off FAIL -> on OK.")
    else:
        print("All rows capture a full graph only after GraphMend.")
    return 1 if failed else 0


if __name__ == "__main__":
    _paths.check()
    sys.exit(main(sys.argv[1:]))

"""GraphMend paper reproduction harness (break-elimination + output correctness).

For each registered model, runs the forward pass through a counting backend in
two isolated subprocesses (GraphMend off, then on), and reports:
  - breaks before / after  (reproduces Table 2 break counts + fix rate)
  - output fingerprint match (reproduces the bit-identical FP32 claim)

Both arms go through `jac run`, differing only in the `[run] graphmend` config,
so GraphMend is the single variable between them. Running the entry under plain
CPython instead would silently disable every [Defer] rewrite -- see entry.py.

CPU-reproducible; GPU is only needed for the latency/throughput numbers, which
are out of scope here. Run with the repo jaclang on PYTHONPATH:

    PYTHONPATH=$PWD python -m paper_eval.run_eval [model_key ...]
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

from paper_eval.registry import MODELS, NETWORK_MODELS

_JAC_TOML = """\
[run]
graphmend = {on}
graphmend_claim_imports = {on}
"""


def _run(key: str, mode: str, state: str = "") -> dict:
    """Run one model in one mode under `jac run`, in its own working directory.

    The mode is carried by a jac.toml written next to the entry program rather
    than by a flag: there is no CLI switch for GraphMend, and `${VAR}`
    interpolation yields strings where these keys need real booleans.

    `state` is a path the two arms use to hand weights from the first to the
    second, so the comparison is between two runs of the same model rather than
    between two random initialisations that are only assumed to agree. See the
    GM_STATE block in entry.py.
    """
    repo = os.getcwd()
    entry_src = os.path.join(repo, "paper_eval", "entry.py")
    workdir = tempfile.mkdtemp(prefix=f"gm_{key.replace('/', '_')}_{mode}_")
    try:
        with open(os.path.join(workdir, "jac.toml"), "w") as fh:
            fh.write(_JAC_TOML.format(on="true" if mode == "on" else "false"))
        shutil.copy(entry_src, os.path.join(workdir, "entry.py"))

        env = dict(os.environ, PYTHONPATH=repo, PAPER_EVAL_DIR=repo, GM_MODEL=key)
        if state:
            env["GM_STATE"] = state
        p = subprocess.run(
            [sys.executable, "-m", "jaclang", "run", "entry.py"],
            capture_output=True, text=True, env=env, cwd=workdir,
        )
        for line in reversed(p.stdout.strip().splitlines()):
            if line.startswith("GMRESULT "):
                row = json.loads(line[len("GMRESULT "):])
                row["mode"] = mode
                return row
        # Lead with the exception line and the exit code rather than a bare
        # tail. A 200-character tail of a run that died mid-compile is just
        # compiler progress output, which says nothing about the cause, and a
        # SIGKILLed arm writes no traceback at all: the OOM killer takes the
        # process while GraphMend is compiling the imported modeling code, and
        # the row then reports ERR for what is really a memory ceiling.
        err = (p.stderr.strip() or p.stdout.strip())
        head = ""
        if p.returncode in (-9, 137):
            head = (f"killed by SIGKILL (exit {p.returncode}), almost certainly "
                    "out of memory: give Docker at least 10 GB, see the "
                    "troubleshooting section of artifact/README.md")
        else:
            head = next(
                (s for s in (ln.strip() for ln in reversed(err.splitlines()))
                 if re.match(r"^[A-Za-z_][\w.]*(Error|Exception|Interrupt):", s)),
                "")
        tail = err[-200:]
        return {"key": key, "mode": mode, "graphs": None, "breaks": None,
                "out_hash": None, "in_shape": None,
                "error": (head + " || " if head else "")
                         + (tail or f"no GMRESULT (exit {p.returncode})")}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main(keys):
    rows = []
    tot_before = tot_after = 0
    for key in keys:
        print(f"  ... {key}", file=sys.stderr, flush=True)
        # One scratch file per model, removed straight after: the off arm
        # writes its weights into it and the on arm loads them.
        state_dir = tempfile.mkdtemp(prefix=f"gmstate_{key.replace('/', '_')}_")
        state = os.path.join(state_dir, "weights.pt")
        try:
            off = _run(key, "off", state)
            on = _run(key, "on", state)
        finally:
            shutil.rmtree(state_dir, ignore_errors=True)
        err = off["error"] or on["error"]
        if err:
            rows.append((key, "-", "-", "-", "ERR", "-"))
            print(f"  {key}: {err}", file=sys.stderr)
            continue
        b0, b1 = off["breaks"], on["breaks"]
        fixed = b0 - b1
        pct = f"{100 * fixed // b0 if b0 else 100}%"
        # THREE-state, not two. `None == None` is True, so folding the
        # "no fingerprint available" case in here reported `yes` for a row
        # that compared nothing, which is a false pass on the correctness
        # half of the claim.
        if off["out_hash"] is None or on["out_hash"] is None:
            correct = "n/a"
        else:
            correct = "yes" if off["out_hash"] == on["out_hash"] else "NO"
        # The off/on runs must use identical inputs for the comparison to mean
        # anything; show the shape and flag a mismatch rather than trusting it.
        s0, s1 = off.get("in_shape"), on.get("in_shape")
        shape = "x".join(str(d) for d in (s0 or [])) or "-"
        if s0 != s1:
            shape += " MISMATCH"
        tot_before += b0
        tot_after += b1
        rows.append((key, b0, b1, pct, correct, shape))

    print(f"\n{'model':28} {'breaks_before':>13} {'breaks_after':>12} "
          f"{'fixed':>6} {'output_ok':>9} {'input':>12}")
    print("-" * 85)
    for r in rows:
        print(f"{r[0]:28} {str(r[1]):>13} {str(r[2]):>12} "
              f"{str(r[3]):>6} {str(r[4]):>9} {str(r[5]):>12}")
    print("-" * 85)
    if tot_before:
        print(f"{'TOTAL':28} {tot_before:>13} {tot_after:>12} "
              f"{100 * (tot_before - tot_after) // tot_before:>5}% "
              f"(eliminated {tot_before - tot_after}/{tot_before})")


if __name__ == "__main__":
    # Named models run as asked; a bare run skips the ones needing network so the
    # default stays offline and download-free.
    args = sys.argv[1:]
    if not args:
        args = [k for k in MODELS if k not in NETWORK_MODELS]
        skipped = sorted(NETWORK_MODELS)
        if skipped:
            print(f"(skipping network models: {', '.join(skipped)} -- "
                  f"run by name to include)", file=sys.stderr)
    main(args)

"""GraphMend paper reproduction harness (break-elimination + output correctness).

For each registered model, runs the forward pass through a counting backend in
two isolated subprocesses (GraphMend off, then on), and reports:
  - breaks before / after  (reproduces Table 2 break counts + fix rate)
  - output fingerprint match (reproduces the bit-identical FP32 claim)

Both arms go through `jac run`, differing only in the `[run] graphmend` config,
so GraphMend is the single variable between them. Running the entry under plain
CPython instead would silently disable every [Defer] rewrite -- see entry.py.

CPU-reproducible; GPU is only needed for the latency/throughput numbers, which
are out of scope here. Run `bash scripts/setup.sh` once, then from the
repository root:

    python -m paper_eval.run_eval [model_key ...]
"""
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile

from paper_eval import _paths
from paper_eval.registry import MODELS, NETWORK_MODELS

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
    entry_src = os.path.join(_paths.ARTIFACT_ROOT, "paper_eval", "entry.py")
    workdir = tempfile.mkdtemp(prefix=f"gm_{key.replace('/', '_')}_{mode}_")
    try:
        with open(os.path.join(workdir, "jac.toml"), "w") as fh:
            fh.write(_JAC_TOML.format(on="true" if mode == "on" else "false",
                                      src=_paths.JACLANG_DIR))
        shutil.copy(entry_src, os.path.join(workdir, "entry.py"))

        env = dict(os.environ, PYTHONPATH=_paths.ARM_PYTHONPATH,
                   PAPER_EVAL_DIR=_paths.ARTIFACT_ROOT, GM_MODEL=key)
        if state:
            env["GM_STATE"] = state
        # A hung arm has to fail with a reason rather than stall the sweep in
        # silence. start_new_session puts the arm in its own process group so
        # the compiler child dies with it; killing only the direct child leaves
        # it running and holding the memory.
        limit = float(os.environ.get("GM_ARM_TIMEOUT", "5400"))
        proc = subprocess.Popen(
            [sys.executable, "-m", "jaclang", "run", "entry.py"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=env, cwd=workdir, start_new_session=True)
        try:
            out, errtxt = proc.communicate(timeout=limit)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                proc.kill()
            proc.communicate()
            return {"key": key, "mode": mode, "graphs": None, "breaks": None,
                    "out_hash": None, "in_shape": None,
                    "error": f"timed out after {int(limit)}s with GraphMend "
                             f"{mode}. Raise GM_ARM_TIMEOUT to allow longer."}
        for line in reversed(out.strip().splitlines()):
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
        err = (errtxt.strip() or out.strip())
        head = ""
        if rc in (-9, 137):
            head = (f"killed by SIGKILL (exit {rc}), almost certainly "
                    "out of memory: give Docker at least 20 GB, see the "
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
                         + (tail or f"no GMRESULT (exit {rc})")}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# Rows far heavier than the rest. Measured LAST, so a row that exhausts memory
# or has to be abandoned cannot block the ones behind it.
RUN_LAST = ("chronos-bolt-small",)


def main(keys):
    rows = []
    tot_before = tot_after = 0
    # Name the columns before the rows start arriving. The rows stream one at a
    # time and the consolidated table only prints at the end, so without this a
    # reviewer is reading six unlabelled values for the length of the run.
    print(f"ROW {'model':28} {'breaks':>8} {'after':>8} {'fixed':>7} "
          f"{'correctness':>13}", flush=True)
    print(f"    {'-' * 68}", flush=True)
    # Stable: the heavy rows move to the end, everything else keeps its order.
    keys = sorted(keys, key=lambda k: k in RUN_LAST)
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
            print(f"ROW {key:28} {'-':>8} {'-':>8} {'-':>7} {'ERR':>13}",
                  flush=True)
            print(f"  {key}: {err}", file=sys.stderr)
            continue
        b0, b1 = off["breaks"], on["breaks"]
        fixed = b0 - b1
        pct = f"{100 * fixed // b0 if b0 else 100}%"
        correct = "yes" if off["out_hash"] == on["out_hash"] else "NO"
        # The off/on runs must use identical inputs for the comparison to mean
        # anything; show the shape and flag a mismatch rather than trusting it.
        s0, s1 = off.get("in_shape"), on.get("in_shape")
        shape = "x".join(str(d) for d in (s0 or [])) or "-"
        if s0 != s1:
            # The comparison only means something if both arms saw the same
            # input, so say so loudly rather than leaving it to be noticed in a
            # column.
            shape += " MISMATCH"
            print(f"    {key}: INPUT SHAPE MISMATCH, {s0} against {s1} -- the "
                  f"two arms did not compare the same work", flush=True)
        tot_before += b0
        tot_after += b1
        rows.append((key, b0, b1, pct, correct, shape))
        # Emit the row NOW as well as in the closing table. A sweep is tens of
        # minutes warm and hours cold, and a caller that captures stdout sees
        # nothing at all until the last model finishes -- which makes a working
        # run indistinguishable from a hung one.
        print(f"ROW {key:28} {b0:>8} {b1:>8} {pct:>7} {correct:>13}",
              flush=True)

    # Driven by verify_break_elimination.py, which prints one consolidated
    # table over every row at the end. Printing a second one here just repeats
    # 21 of those rows in a different format.
    if os.environ.get("GM_NO_SUMMARY"):
        return 0 if not any(r[4] == "NO" for r in rows) else 1

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
    _paths.check()
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

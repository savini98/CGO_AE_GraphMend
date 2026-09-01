"""GraphMend's compilation overhead (paper Section 5.7).

    python -m paper_eval.run_overhead [model_key ...]

Times the same entry program end to end under standard Python and under
`jac run`, cold (empty compiler cache) and cached:

    overhead % = (graphmend - baseline) / baseline * 100

Each cold run gets a private XDG_CACHE_HOME, so this neither reads nor destroys
an existing cache. CPU only.
"""

import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

from paper_eval import _paths
from paper_eval.registry import MODELS

# GraphMend on, and claiming imported code, which is what the measured
# pipeline has to do for the comparison to be about GraphMend at all.
_JAC_TOML = """\
[dev]
jaclang_source = "{src}"

[run]
graphmend = true
graphmend_claim_imports = true
"""


def _time_run(key, mode, cache_dir, workdir):
    """One end-to-end run of the entry program. Returns seconds, or None.

    `mode` is "baseline" (plain CPython) or "graphmend" (`jac run`). The two
    differ ONLY in the interpreter path: same file, same inputs, same seed.
    """
    env = dict(
        os.environ,
        PYTHONPATH=_paths.ARM_PYTHONPATH,
        PAPER_EVAL_DIR=_paths.ARTIFACT_ROOT,
        GM_MODEL=key,
        XDG_CACHE_HOME=cache_dir,
    )
    if mode == "baseline":
        cmd = [sys.executable, "entry.py"]
    else:
        cmd = [sys.executable, "-m", "jaclang", "run", "entry.py"]

    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=workdir)
    elapsed = time.perf_counter() - t0
    if p.returncode != 0:
        return None, (p.stderr.strip() or p.stdout.strip())[-300:]
    return elapsed, None


def measure(key):
    """Cold and cached overhead for one model, as percentages over baseline."""
    workdir = tempfile.mkdtemp(prefix=f"gm_ovh_{key.replace('/', '_')}_")
    cache = tempfile.mkdtemp(prefix="gm_ovh_cache_")
    try:
        shutil.copy(os.path.join(_paths.ARTIFACT_ROOT, "paper_eval", "entry.py"),
                    os.path.join(workdir, "entry.py"))
        with open(os.path.join(workdir, "jac.toml"), "w") as fh:
            fh.write(_JAC_TOML.format(src=_paths.JACLANG_DIR))

        # Baseline first, and twice: the first CPython run pays for importing
        # torch and transformers into a cold OS page cache, which is not a
        # GraphMend cost and would otherwise be charged to the baseline and so
        # credited to GraphMend as negative overhead.
        base_warm, err = _time_run(key, "baseline", cache, workdir)
        if base_warm is None:
            return {"key": key, "error": err}
        base, err = _time_run(key, "baseline", cache, workdir)
        if base is None:
            return {"key": key, "error": err}

        # Cold: the private cache is empty, so every pass runs and the Jac
        # compiler's own sources compile too.
        cold, err = _time_run(key, "graphmend", cache, workdir)
        if cold is None:
            return {"key": key, "error": err}

        # Cached: same cache directory, now populated.
        cached, err = _time_run(key, "graphmend", cache, workdir)
        if cached is None:
            return {"key": key, "error": err}

        return {
            "key": key,
            "baseline_s": base,
            "cold_s": cold,
            "cached_s": cached,
            "cold_pct": (cold - base) / base * 100.0,
            "cached_pct": (cached - base) / base * 100.0,
            "error": None,
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        shutil.rmtree(cache, ignore_errors=True)


def main(keys):
    keys = keys or [k for k, s in MODELS.items() if not s.get("network")]
    print(f"{'model':<32}{'baseline':>10}{'cold':>10}{'cached':>10}"
          f"{'cold %':>10}{'cached %':>10}")
    print("-" * 82)
    rows = []
    for key in keys:
        print(f"  {key} ...", file=sys.stderr, flush=True)
        r = measure(key)
        rows.append(r)
        if r.get("error"):
            print(f"{key:<32}{'ERR':>10}   {r['error'][:40]}")
            continue
        print(f"{r['key']:<32}{r['baseline_s']:>9.2f}s{r['cold_s']:>9.2f}s"
              f"{r['cached_s']:>9.2f}s{r['cold_pct']:>9.1f}%{r['cached_pct']:>9.1f}%")

    ok = [r for r in rows if not r.get("error")]
    if ok:
        print("-" * 82)
        cold = statistics.mean(r["cold_pct"] for r in ok)
        cached = statistics.mean(r["cached_pct"] for r in ok)
        fixed = statistics.mean(r["cold_s"] - r["baseline_s"] for r in ok)
        print(f"{'MEAN over ' + str(len(ok)) + ' models':<32}"
              f"{'':>10}{'':>10}{'':>10}{cold:>9.1f}%{cached:>9.1f}%")
        print()
        print(f"Paper Section 5.7 reports a mean of 11.5% cold and 1.1% cached.")
        print(f"Mean absolute cold cost here: {fixed:.2f}s per model "
              f"(the paper's one-time ~0.5s charge, which is why the "
              f"percentage falls on longer workloads).")
    if os.environ.get("GM_OVERHEAD_JSON"):
        with open(os.environ["GM_OVERHEAD_JSON"], "w") as fh:
            json.dump(rows, fh, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    _paths.check()
    sys.exit(main(sys.argv[1:]))

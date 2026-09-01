#!/usr/bin/env python3
"""Claim 2: the transformation's effect on cold-start and steady-state latency.

    python artifact/claim2_latency.py                 # every model with fixed sources
    python artifact/claim2_latency.py t5-small        # one row
    python artifact/claim2_latency.py --list          # what would run, and why not

CLAIM 1 (artifact/table2_breaks.py) proves GraphMend eliminates the breaks and
preserves output. It has to run the compiler, because the compiler is the
contribution. THIS script measures what that transformation buys, and it does
NOT run the compiler: both arms are plain CPython importing plain Python.

    original arm   stock transformers
    fixed arm      the same tree with GraphMend's OUTPUT copied over the
                   modules it transformed, from artifact/fixed_models/

That is the paper's own methodology -- the reference scripts compare stock
transformers against fixed model files and never recompile during a timed run --
and it is why those runs land on Table 2 where compiling inside the measured
window does not: the front end's minutes stop falling inside the profiled region.
It also makes each arm seconds rather than minutes, which is the difference
between a reviewer running one row and running all of them.

PROVENANCE. The fixed sources are compiler output, not hand edits, and that is
checkable rather than asserted: artifact/gen_fixed_models.py regenerates them
from the installed transformers, and every file ships beside its `.original.py`.
Regenerate and diff if you do not want to take it on trust.

WHAT TO EXPECT. Cold start in the single to low double digits, and inversely
related to batch size: the same model reads 3.49x at batch 1345 and 15.66x at
8x128. Steady state within a few percent of 1.0x. Not compared to Table 2 row by
row, since the paper's figures hold at its own batch sizes.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

# Table 2, RTX 3090: (cold, steady). Keys are bench.py model keys.
TABLE2 = {
    "rebel-large": (19.86, 1.12),
    "bart-large-cnn": (21.07, 1.13),
    "opus-mt-fr-en": (13.16, 1.10),
    "bart-base": (11.87, 1.15),
    "t5-small": (3.49, 1.08),
    "MoLFormer-XL-both10pct": (24.71, 1.13),
    "t5-base": (2.27, 1.09),
    "t5-3b": (3.01, 1.08),
    "flan-t5-large": (3.27, 1.06),
    "inclusively-reformulation-it5": (3.02, 1.07),
    "biogpt": (2.63, 1.11),
    "blenderbot-400M-distill": (3.21, 1.12),
    "tiny-random-PegasusForCausalLM": (2.72, 1.34),
    "longformer-scico": (2.53, 1.05),
    "whisper-base": (2.49, 1.09),
    "whisper-small": (3.09, 1.08),
    "whisper-large-v3": (3.06, 1.08),
    "grounding-dino-tiny": (5.20, 1.08),
    "grounding-dino-base": (5.17, 1.06),
    "layoutlmv3-base": (6.78, 1.06),
    "chronos-bolt-small": (4.64, 1.09),
    "Florence-2": (20.95, 1.19),
    "Qwen-Audio-Chat": (2.76, 1.15),
    "Phi-4-mini-instruct": (3.60, 1.13),
}

# Which fixed_models/<dir> supplies each row's transformed sources. Several rows
# share one: the BART family is three models over one modeling file.
SOURCES = {
    "t5-small": ["t5-small"],
    "MoLFormer-XL-both10pct": ["MoLFormer-XL-both10pct"],  # hub remote code
    "Phi-4-mini-instruct": ["Phi-4-mini-instruct"],
    "bart-large-cnn": ["bart"],
    "bart-base": ["bart"],
    "rebel-large": ["bart"],
    "opus-mt-fr-en": ["opus-mt-fr-en"],
    # dump dirs are keyed by REGISTRY name for three of these
    "t5-base": ["t5-base"],
    "t5-3b": ["t5-3b"],
    "flan-t5-large": ["flan-t5-large"],
    "inclusively-reformulation-it5": ["inclusively-reformulation-it5"],
    "biogpt": ["biogpt"],
    "blenderbot-400M-distill": ["blenderbot-400M-distill"],
    "tiny-random-PegasusForCausalLM": ["PegasusForCausalLM"],
    "longformer-scico": ["longformer-base-4096"],
    "whisper-base": ["whisper-base"],
    "whisper-small": ["whisper-small"],
    "whisper-large-v3": ["whisper"],
    "grounding-dino-tiny": ["grounding-dino"],
    "grounding-dino-base": ["grounding-dino-base"],
    "layoutlmv3-base": ["layoutlmv3-base"],
    "chronos-bolt-small": ["chronos-bolt-small"],
    "Florence-2": ["Florence-2"],
    "Qwen-Audio-Chat": ["Qwen-Audio-Chat"],
}

# The paper's per-model batch, read from run_all_3090_new.log in the research
# repository. Cold start is strongly batch-sensitive -- the same model reads
# 3.49x at batch 1345 and 15.66x at 8x128 -- so a magnitude is only comparable
# to Table 2 when the batch is.
#
# ONE VALUE PER MODEL, USED FOR BOTH ARMS. The reference auto-detects batch from
# free GPU memory, and the fixed model needs less of it, so five rows were
# measured with the arms on DIFFERENT batches (t5-small 1345/1361, opus-mt
# 1089/1270, t5-base 350/407, t5-3b 20/54, it5 105/188). A ratio across unequal
# batches compares unequal work. The original arm's batch is the one taken here,
# because that is the size the unmodified model was measured at.

# A 10-row subset for reviewers who do not want to spend an hour on the full
# sweep. Chosen for coverage rather than speed alone: every rule fires at least
# once ([Trap] on MoLFormer and grounding-dino-base, [Where] on Phi-4-mini and
# Qwen-Audio-Chat, [Defer] on the rest), every input modality appears (text,
# speech, vision, time series), and both ends of the agreement range are here:
# bart-large-cnn and rebel-large land near Table 2, grounding-dino-base and
# MoLFormer are the two furthest from it.
QUICK_ROWS = [
    "rebel-large",
    "bart-large-cnn",
    "opus-mt-fr-en",
    "t5-small",
    "MoLFormer-XL-both10pct",
    "whisper-base",
    "chronos-bolt-small",
    "grounding-dino-base",
    "Qwen-Audio-Chat",
    "Phi-4-mini-instruct",
]

PAPER_BATCH = {
    "t5-small": 1345,
    "bart-base": 811,
    "bart-large-cnn": 159,
    "rebel-large": 180,
    "opus-mt-fr-en": 1089,
    "MoLFormer-XL-both10pct": 837,
}

COLD_TOL = float(os.environ.get("GM_COLD_TOL", "0.35"))
WARM_TOL = float(os.environ.get("GM_WARM_TOL", "0.10"))

# [Defer] rewrites a logger call to __jac_log_emit__, which only BUFFERS while a
# depth counter is raised -- and that counter is raised by a forward pre-hook
# GraphMend injects at the `torch.compile(...)` site in the entry program, not in
# the modeling file. A fixed source dropped into plain CPython therefore emits
# inline and keeps its logger break. bench.py's arm builds the compiled module
# itself, so the hooks are registered here through a sitecustomize shim that
# wraps torch.compile once, for the fixed arm only.
SHIM = '''
# Registers the [Defer] buffer hooks on anything torch.compile returns. Without
# this a [Defer]-fixed source runs its logger calls inline and the break stays.
import torch

try:
    from jaclang.lib.jaclib import __jac_se_region_open__, __jac_log_flush_hook__
except Exception:
    __jac_se_region_open__ = __jac_log_flush_hook__ = None

if __jac_se_region_open__ is not None:
    _orig_compile = torch.compile

    def _compile_with_defer_hooks(*a, **k):
        compiled = _orig_compile(*a, **k)
        if hasattr(compiled, "register_forward_pre_hook"):
            compiled.register_forward_pre_hook(__jac_se_region_open__)
            compiled.register_forward_hook(__jac_log_flush_hook__,
                                           always_call=True)
        return compiled

    torch.compile = _compile_with_defer_hooks
'''



def apply_row_sources(dest_root, fixed_models, row):
    """Re-assert one row's fixed sources over the shared tree.

    Returns the basenames written. Rows sharing a filename (ten ship
    modeling_utils.py, in four different versions) would otherwise run whichever
    copy was written last, which is a silent wrong-model measurement rather than
    an error.
    """
    dest = os.path.join(dest_root, "transformers")
    written = []
    for srcdir in SOURCES.get(row, []):
        d = os.path.join(fixed_models, srcdir)
        for fixed in sorted(glob.glob(os.path.join(d, "*.graphmend.py"))):
            base = os.path.basename(fixed)[:-len(".graphmend.py")]
            hits = [p for p in glob.glob(os.path.join(dest, "**", base + ".py"),
                                         recursive=True)]
            if hits:
                shutil.copyfile(fixed, hits[0])
                written.append(base)
    return written


def build_fixed_tree(fixed_models, rows, workdir, python):
    """A transformers tree with GraphMend's output copied over the originals.

    Copied rather than patched in place: a timed run must not mutate the
    interpreter the original arm is using, and a killed run must not leave a
    half-patched site-packages behind.
    """
    probe = subprocess.run(
        [python, "-c", "import transformers,os;print(os.path.dirname(transformers.__file__))"],
        capture_output=True, text=True, check=False)
    stock = probe.stdout.strip()
    if not stock or not os.path.isdir(stock):
        raise RuntimeError("cannot locate the installed transformers")
    dest_root = os.path.join(workdir, "fixed_site")
    dest = os.path.join(dest_root, "transformers")
    if not os.path.isdir(dest):
        shutil.copytree(stock, dest, symlinks=True,
                        ignore=shutil.ignore_patterns("__pycache__"))

    applied, defer_only = [], []
    for row in rows:
        for srcdir in SOURCES.get(row, []):
            d = os.path.join(fixed_models, srcdir)
            for fixed in sorted(glob.glob(os.path.join(d, "*.graphmend.py"))):
                base = os.path.basename(fixed)[: -len(".graphmend.py")]
                # Place it where that module lives in the tree.
                hits = [p for p in glob.glob(os.path.join(dest, "**", base + ".py"),
                                             recursive=True)]
                if not hits:
                    continue
                shutil.copyfile(fixed, hits[0])
                applied.append((srcdir, base))
                body = open(fixed, encoding="utf-8").read()
                if "__jac_log_emit__" in body and "__gm_cond_" not in body:
                    defer_only.append(base)
    # Hub remote code (MoLFormer, Florence-2, Qwen) lives in the HF modules
    # cache, not in the transformers package, so patching only the package
    # leaves those rows running the stock model in BOTH arms -- the signature
    # is identical launch counts, which the caller fails on. Copy and redirect
    # rather than edit, so the original arm and the real cache stay untouched.
    # Resolve the modules cache the way transformers does, not from ~. In a
    # container HF_HOME is set (/hf here) while ~ is /root, so a hardcoded
    # ~/.cache/huggingface/modules names a directory that does not exist: the
    # copy silently finds nothing, the Hub rows keep their unpatched code, and
    # MoLFormer fails the launch gate at 50 -> 50 as if the transform had done
    # nothing. Florence-2 and Qwen-Audio-Chat load the same way.
    _hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    mod_cache = (os.environ.get("HF_MODULES_CACHE")
                 or os.path.join(_hf_home, "modules"))
    mod_copy = os.path.join(workdir, "hf_modules")
    if os.path.isdir(mod_cache):
        if not os.path.isdir(mod_copy):
            shutil.copytree(mod_cache, mod_copy, symlinks=True,
                            ignore=shutil.ignore_patterns("__pycache__"))
        for row in rows:
            for srcdir in SOURCES.get(row, []):
                d = os.path.join(fixed_models, srcdir)
                for fixed in sorted(glob.glob(os.path.join(d, "modeling_*.graphmend.py"))):
                    base = os.path.basename(fixed)[: -len(".graphmend.py")]
                    for hit in glob.glob(os.path.join(mod_copy, "**", base + ".py"),
                                         recursive=True):
                        shutil.copyfile(fixed, hit)
                        applied.append((srcdir, base + " (hub)"))
    else:
        mod_copy = None

    fh = open(os.path.join(dest_root, "sitecustomize.py"), "w", encoding="utf-8")
    fh.write(SHIM)
    fh.close()
    return dest_root, applied, defer_only, mod_copy


def run_arm(bench, python, key, on, trace_dir, extra_path, runs, hf_modules=None):
    """One arm, in-process in bench.py, with no compiler in the loop."""
    env = dict(os.environ)
    env["GM_BENCH10_ARM"] = "1"
    env["GM_BENCH10_ON"] = "1" if on else "0"
    env["GM_BENCH10_RUNS"] = str(runs)
    env["GM_BENCH10_TRACE_DIR"] = trace_dir
    env["PYTHONUNBUFFERED"] = "1"
    # The fixed arm imports the patched tree; the shim beside it registers the
    # [Defer] hooks. The original arm sees neither.
    if extra_path:
        env["PYTHONPATH"] = extra_path + os.pathsep + env.get("PYTHONPATH", "")
    if hf_modules:
        env["HF_MODULES_CACHE"] = hf_modules
    env["GM_MODEL"] = key
    # Same batch for BOTH arms, from the paper's own log. The reference
    # auto-detects it from free GPU memory, and the fixed model needs less, so
    # five rows there were measured with the arms on different batches; a ratio
    # across unequal batches compares unequal work.
    # GM_PAPER_BATCH=1 pins the paper's per-model batch (from its own run log)
    # for BOTH arms. Off by default: bench.py's small defaults are what a
    # reviewer gets without the paper's internal log, they run on any card, and
    # the paper's sizes (t5-small 1345) do not fit a 24 GB 3090 with anything
    # else resident. The trade-off is direction, not validity: a small batch
    # leaves the fixed compile cost unamortised, so cold start reads HIGH, while
    # the paper's larger batches read lower. Table 2 sits between the two.
    if os.environ.get("GM_PAPER_BATCH") and key in PAPER_BATCH:
        env["GM_BENCH10_BATCH"] = str(PAPER_BATCH[key])
    # A PRIVATE Inductor cache per arm. Sharing one lets the second arm reuse
    # kernels the first compiled and inflates the cold ratio several-fold.
    icache = os.path.join(trace_dir, "inductor_" + ("on" if on else "off"))
    os.makedirs(icache, exist_ok=True)
    env["TORCHINDUCTOR_CACHE_DIR"] = icache
    p = subprocess.run([python, bench], capture_output=True, text=True,
                       env=env, check=False)
    for line in p.stdout.splitlines():
        if line.startswith("GMBENCH10 "):
            return json.loads(line[len("GMBENCH10 "):])
    tail = (p.stderr.strip().splitlines() or ["no GMBENCH10 line"])[-1]
    raise RuntimeError(tail[:160])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*")
    ap.add_argument("--fixed-models", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "fixed_models"))
    ap.add_argument("--runs", default="10",
                    help="forward passes per arm (default 10): the first is cold, "
                         "the rest are medianed for steady state")
    ap.add_argument("--repeat", type=int, default=5,
                    help="independent measurements per row (default 5); cold "
                         "start is one sample per measurement, so repeats are "
                         "what average it")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="the 10-row subset in QUICK_ROWS, about a "
                         "quarter of the time of the full sweep")
    o = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    bench = os.path.join(here, "gpu", "bench.py")
    rows = o.models or (QUICK_ROWS if o.quick else list(TABLE2))
    rows = [r for r in rows if r in TABLE2]

    have = {r for r in rows
            if any(glob.glob(os.path.join(o.fixed_models, s, "*.graphmend.py"))
                   for s in SOURCES.get(r, []))}
    if o.list:
        for r in rows:
            print(f"{r:28s} {'ready' if r in have else 'no fixed sources'}")
        return 0
    if not have:
        sys.exit(f"no fixed sources under {o.fixed_models}; "
                 f"run artifact/gen_fixed_models.py first")

    work = tempfile.mkdtemp(prefix="claim2_")
    fixed_path, applied, defer_only, hfmod = build_fixed_tree(
        o.fixed_models, sorted(have), work, sys.executable)
    print(f"fixed sources applied: {len(applied)}")
    for srcdir, base in applied:
        print(f"    {srcdir}/{base}")
    if defer_only:
        print(f"  [Defer]-only files present ({', '.join(sorted(set(defer_only)))});"
              f" the hook shim is active for the fixed arm")
    print()

    results, bad = [], 0
    for key in rows:
        if key not in have:
            print(f"{key:28s} SKIP (no fixed sources)")
            continue
        tdir = os.path.join(work, key)
        os.makedirs(tdir, exist_ok=True)
        reps, failed = [], None
        for _rep in range(max(1, o.repeat)):
            try:
                off = run_arm(bench, sys.executable, key, False, tdir, None,
                              o.runs)
                # The shared tree may hold another row's copy of a same-named
                # module by now, so re-assert this row's own sources first.
                apply_row_sources(fixed_path, o.fixed_models, key)
                on = run_arm(bench, sys.executable, key, True, tdir, fixed_path,
                             o.runs, hfmod)
            except RuntimeError as exc:
                failed = str(exc)
                break
            reps.append((off, on))
        if failed is not None:
            print(f"{key:28s} FAIL ({failed})")
            bad += 1
            continue
        off, on = reps[-1]

        if os.environ.get("GM_DEBUG_ARMS"):
            print(f"  RAW {key} stock {json.dumps(off, sort_keys=True)}")
            print(f"  RAW {key} fixed {json.dumps(on, sort_keys=True)}")
        lo, ln = off.get("cudagraph_launches"), on.get("cudagraph_launches")
        if lo is None or ln is None:
            print(f"{key:28s} FAIL (no launch count)")
            bad += 1
            continue
        if lo == ln:
            # The one failure that otherwise looks like a clean 1.00x result.
            print(f"{key:32s} FAIL (CUDA-graph launches unchanged at {lo}: the "
                  f"fixed sources never reached the compiled program)")
            bad += 1
            continue
        try:
            # The raw first-iteration window, deliberately, and NOT the
            # compile-excluded cold_ms. cold_start_no_compile.py subtracts
            # events named "backend_compile", and there are zero of those in
            # any of the paper's own reference traces -- torch records that work
            # as "dynamo_timed" spans instead. So on the traces the paper was
            # computed from, its compile_ms is 0 and its `cold_speedup` equals
            # the `old_speedup` it meant to replace. The raw window IS the
            # published metric. Using cold_ms here would silently diverge on any
            # torch that does emit backend_compile.
            # Ratio per measurement, then averaged. Averaging the two arms'
            # times separately would mix machine conditions across runs; the
            # pairing inside one measurement is what cancels them.
            colds = [a["cold_window_ms"] / b["cold_window_ms"] for a, b in reps]
            warms = [a["warm_ms"] / b["warm_ms"] for a, b in reps]
            cold = sum(colds) / len(colds)
            warm = sum(warms) / len(warms)
        except (KeyError, ZeroDivisionError):
            print(f"{key:28s} FAIL (incomplete timing)")
            bad += 1
            continue
        spread = ((max(colds) - min(colds)) / cold * 100) if len(colds) > 1 else 0.0
        results.append((key, cold, warm, lo, ln, spread, len(colds)))

    print()
    print(f"{'model':32s} {'cold':>10s} {'steady state':>14s}")
    print("-" * 60)
    for key, cold, warm, lo, ln, _sp, _n in results:
        print(f"{key:32s} {cold:9.2f}x {warm:13.3f}x")
    print("-" * 60)
    if results and results[0][6] > 1:
        worst = max(results, key=lambda r: r[5])
        print(f"each value is the mean of {results[0][6]} measurements; "
              f"widest cold spread {worst[5]:.0f}% on {worst[0]}")
    print(f"{len(results)} row(s) measured on this GPU, from traces this run")
    print("produced. Expect cold start in the single to low double digits")
    print("(inversely related to batch) and steady state near 1.0x.")
    if bad:
        print(f"\n{bad} row(s) failed.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

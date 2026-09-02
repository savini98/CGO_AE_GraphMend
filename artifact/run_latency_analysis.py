#!/usr/bin/env python3
"""C3: the effect of removing graph breaks on cold-start and steady-state latency.

    python artifact/run_latency_analysis.py            # the 10-model sample
    python artifact/run_latency_analysis.py --full     # every model
    python artifact/run_latency_analysis.py t5-small   # one model
    python artifact/run_latency_analysis.py --list     # what would run

Neither arm runs the compiler. The original arm is stock transformers; the
fixed arm is the same tree with the modules GraphMend transformed replaced by
its output, from artifact/fixed_models/. Regenerate those with
artifact/gen_fixed_models.py and gate them with artifact/verify_fixed.py.

Needs a CUDA device, and downloads pretrained weights for every model.
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

# The paper's per-model batch. Cold start is batch-sensitive, so a magnitude
# is only comparable to the paper's when the batch is.
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

# Sequence length per row, from each reference script's fixed_batch signature.
# See the note above: the batch is only half the workload.
PAPER_SEQ = {
    "t5-small": 5, "t5-base": 5, "t5-3b": 5, "flan-t5-large": 5,
    "inclusively-reformulation-it5": 5, "bart-base": 5, "bart-large-cnn": 5,
    "rebel-large": 5, "opus-mt-fr-en": 5, "biogpt": 5,
    "blenderbot-400M-distill": 5, "tiny-random-PegasusForCausalLM": 5,
    "longformer-scico": 5, "Qwen-Audio-Chat": 5, "Phi-4-mini-instruct": 5,
    "chronos-bolt-small": 128,
}

PAPER_BATCH = {
    "t5-small": 1345,
    "bart-base": 811,
    "bart-large-cnn": 159,
    "rebel-large": 180,
    "opus-mt-fr-en": 1089,
    "MoLFormer-XL-both10pct": 837,
    # The rest. Where the two arms auto-detected
    # differently (opus 1089/1270, it5 105/188, t5-base 350/407) the original
    # arm's value is used: both arms must run one batch or the ratio compares
    # unequal work.
    "t5-base": 350,
    "t5-3b": 20,
    "inclusively-reformulation-it5": 105,
    "whisper-base": 75,
    "whisper-small": 9,
    "whisper-large-v3": 1,
    "layoutlmv3-base": 31,
    "grounding-dino-tiny": 1,
    "grounding-dino-base": 1,
    "Florence-2": 1,
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


def _modules_cache():
    """Where transformers keeps Hub remote code, asked of transformers itself.

    It is derived from HF_HOME, XDG_CACHE_HOME or the home directory, in that
    order, so reconstructing it here gets it wrong wherever the image sets one
    of them. Returns None if transformers cannot be imported.
    """
    probe = subprocess.run(
        [sys.executable, "-c",
         "from transformers.dynamic_module_utils import HF_MODULES_CACHE;"
         "print(HF_MODULES_CACHE)"],
        capture_output=True, text=True, check=False)
    return probe.stdout.strip() or None


def hub_copy(fixed_models, rows, workdir):
    """A patched copy of the Hub modules cache, or (None, []) if there is none yet.

    Rows loaded with trust_remote_code keep their model code in the modules
    cache rather than in the transformers package, so their fixed sources have
    to be written there instead. The cache is created by the first download, so
    on a machine that has never run these models it does not exist when the
    fixed tree is built; this is called again before each fixed arm, by which
    point the original arm has populated it.
    """
    mod_cache = _modules_cache()
    if not mod_cache or not os.path.isdir(mod_cache):
        return None, []
    mod_copy = os.path.join(workdir, "hf_modules")
    if os.path.isdir(mod_copy):
        shutil.rmtree(mod_copy, ignore_errors=True)
    shutil.copytree(mod_cache, mod_copy, symlinks=True,
                    ignore=shutil.ignore_patterns("__pycache__"))
    applied = []
    for row in rows:
        for srcdir in SOURCES.get(row, []):
            d = os.path.join(fixed_models, srcdir)
            for fixed in sorted(glob.glob(os.path.join(d, "modeling_*.graphmend.py"))):
                base = os.path.basename(fixed)[: -len(".graphmend.py")]
                for hit in glob.glob(os.path.join(mod_copy, "**", base + ".py"),
                                     recursive=True):
                    shutil.copyfile(fixed, hit)
                    applied.append((srcdir, base + " (hub)"))
    return mod_copy, applied


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
    mod_copy, hub_applied = hub_copy(fixed_models, rows, workdir)
    applied.extend(hub_applied)

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
    # Same batch for BOTH arms. The reference
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
    # The sequence is fixed by the reference scripts (seq_len=5 for the text
    # rows) and only the BATCH is auto-sized. Applying it in both modes is what
    # makes auto-sizing follow the paper's protocol rather than half of it:
    # auto-batch at our seq 128 agrees with the default,
    # while the paper's 1345x5 gives 10.32x.
    if key in PAPER_SEQ and not os.environ.get("GM_BENCH10_SEQ"):
        env["GM_BENCH10_SEQ"] = str(PAPER_SEQ[key])
    if not os.environ.get("GM_PAPER_BATCH") and not os.environ.get(
            "GM_BENCH10_BATCH"):
        # Fill ~70% of VRAM, the paper's rule, which transfers across GPUs where
        # its recorded numbers do not.
        env["GM_BENCH10_AUTO_BATCH"] = "1"
    if os.environ.get("GM_PAPER_BATCH") and key in PAPER_BATCH:
        env["GM_BENCH10_BATCH"] = str(PAPER_BATCH[key])
        # The sequence has to move with the batch, or the row runs a workload
        # the paper never measured.
        if key in PAPER_SEQ:
            env["GM_BENCH10_SEQ"] = str(PAPER_SEQ[key])
    # A PRIVATE Inductor cache per arm. Sharing one lets the second arm reuse
    # kernels the first compiled and inflates the cold ratio several-fold.
    icache = os.path.join(trace_dir, "inductor_" + ("on" if on else "off"))
    os.makedirs(icache, exist_ok=True)
    env["TORCHINDUCTOR_CACHE_DIR"] = icache
    # Run from a directory whose jac.toml turns claiming OFF. jac resolves
    # configuration to the nearest ancestor and does not merge, so an arm left
    # in the repository root picks up its `graphmend_claim_imports = true`: the
    # shim's `import jaclang` then installs the import hook, every transformers
    # module the arm imports is claimed and recompiled, and the arm this script
    # describes as plain CPython is running the compiler after all -- on top of
    # the fixed sources it was handed.
    armdir = os.path.join(trace_dir, "arm")
    os.makedirs(armdir, exist_ok=True)
    with open(os.path.join(armdir, "jac.toml"), "w") as fh:
        fh.write("[run]\ngraphmend = false\ngraphmend_claim_imports = false\n")
    p = subprocess.run([python, bench], capture_output=True, text=True,
                       env=env, check=False, cwd=armdir)
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
    ap.add_argument("--full", "--stage2", dest="full", action="store_true",
                    help="stage 2: every row. Default is stage 1, the 10-row "
                         "sample, at about a quarter of the time")
    ap.add_argument("--quick", "--stage1", dest="quick", action="store_true",
                    help="stage 1 explicitly (this is the default)")
    o = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    bench = os.path.join(here, "gpu", "bench.py")
    # Stage 1 by default: the full sweep is opt-in, so a reviewer running the
    # script with no arguments gets a result in minutes rather than an hour.
    rows = o.models or (list(TABLE2) if o.full else QUICK_ROWS)
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

    work = tempfile.mkdtemp(prefix="gm_c3_")
    fixed_path, applied, defer_only, hfmod = build_fixed_tree(
        o.fixed_models, sorted(have), work, sys.executable)
    print(f"fixed sources applied: {len(applied)}")
    if os.environ.get("GM_VERBOSE"):
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
                # Pin the fixed arm to whatever the original arm sized to. The
                # fixed model uses less memory, so left to itself it would pick
                # a LARGER batch and the ratio would span unequal work. The
                # reference runs have exactly that flaw on two rows.
                if off.get("auto_batch"):
                    os.environ["GM_BENCH10_BATCH"] = str(off["auto_batch"])
                # The shared tree may hold another row's copy of a same-named
                # module by now, so re-assert this row's own sources first.
                apply_row_sources(fixed_path, o.fixed_models, key)
                # Hub rows keep their model code in the modules cache, which
                # does not exist until something downloads it. On a machine
                # that has never run these models it was absent when the fixed
                # tree was built, so the copy found nothing and the fixed arm
                # ran the stock code in both arms. The original arm above has
                # populated it by now, so build it here instead.
                # Re-copy every time rather than only when the cache was
                # missing: the modules cache is created by the first download,
                # so on a machine that has never run these models it is absent
                # when the fixed tree is built and appears only once the
                # original arm above has run.
                _hfmod, _hub = hub_copy(o.fixed_models, [key], work)
                if _hfmod:
                    hfmod = _hfmod
                    if os.environ.get("GM_VERBOSE"):
                        for _srcdir, _base in _hub:
                            print(f"    {_srcdir}/{_base}", flush=True)
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
        # Report the row as it lands. The table below is only printed once every
        # row is done, and a ten-row run is long enough that a reviewer watching
        # it should not have to wait for the first number.
        if not results[:-1]:
            print(f"{'model':32s} {'cold':>8s} {'steady':>9s} "
                  f"{'launches':>14s}", flush=True)
            print(f"{'-' * 66}", flush=True)
        print(f"{key:32s} {cold:7.2f}x {warm:8.3f}x "
              f"{str(lo) + ' -> ' + str(ln):>14s}", flush=True)

    print()
    print(f"{'model':32s} {'cold':>10s} {'steady state':>14s} "
          f"{'launches':>14s}")
    print("-" * 75)
    for key, cold, warm, lo, ln, _sp, _n in results:
        print(f"{key:32s} {cold:9.2f}x {warm:13.3f}x "
              f"{str(lo) + ' -> ' + str(ln):>14s}")
    print("-" * 75)
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

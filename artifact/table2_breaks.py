"""Does GraphMend eliminate graph breaks without changing the output?

    python artifact/table2_breaks.py             # every model
    python artifact/table2_breaks.py t5-small    # a subset, for a quick check
    python artifact/table2_breaks.py --offline   # skip rows needing downloads

Runs every model twice, GraphMend off then on, and reports what the transform
removed and whether the result survived it. Exit status is non-zero if any row
fails to run or changes its output.

The output comparison is the load-bearing half. Eliminating a graph break while
altering the result is not a fix, so a row whose two arms disagree fails
outright rather than being reported as a successful reduction.

Both arms go through `jac run` with a jac.toml differing only in
`graphmend_claim_imports`, so what is measured is the compiler's own
transformation of imported model code, not a hand-edited model file. See
`jac/paper_eval/README.md` for why the entry program has to be Jac-compiled.

WHY ROWS TAKE DIFFERENT PATHS. A break count is a property of the code
TorchDynamo actually traces, so it depends on how a model is built and what it
is fed. Most rows are insensitive to that and the small random-weight harness
in `jac/paper_eval/` measures them directly. Five are sensitive and use a
reference-fidelity build in `gpu/bench.py` instead, each for a reason found by
reading the reference scripts rather than guessed:

  * BART family (bart-base, bart-large-cnn, rebel-large, opus-mt-fr-en). The
    guard at modeling_bart.py:568 leads with `dtype == torch.float16`, a static
    Python bool that Dynamo folds to False in fp32, so the data-dependent
    breaks do not exist there: 3 instead of 7. The batch matters independently,
    an `attention_mask` and a decoder input of length ONE.

  * grounding-dino. Not dtype, which the reference pins to fp32 deliberately.
    It needs the real config, a batch built by the model's own processor from a
    real image, and `dynamic` left at its default.

NEITHER NEEDS A GPU. What sets those counts is dtype and input, not the device;
measured on CPU, bart-base reads 3 breaks in fp32 and 7 in fp16.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

# Table 2: (breaks, fix rate %). Keys are this artifact's model names.
TABLE2 = {
    "t5-small": (3, 100), "t5-base": (3, 100), "t5-3b": (3, 100),
    "flan-t5-large": (3, 100), "inclusively-reformulation-it5": (3, 100),
    "whisper": (3, 100), "whisper-small": (3, 100), "whisper-base": (3, 100),
    "bart": (7, 100), "bart-base": (7, 100), "rebel-large": (7, 100),
    "opus-mt-fr-en": (6, 100), "biogpt": (2, 100),
    "blenderbot-400M-distill": (3, 100), "PegasusForCausalLM": (2, 100),
    "layoutlmv3-base": (2, 100), "Phi-4-mini-instruct": (5, 100),
    "grounding-dino": (17, 58), "grounding-dino-base": (17, 58),
    "longformer-base-4096": (5, 40), "clap-htsat-fused": (4, 0),
    "chronos-bolt-small": (6, 100), "MoLFormer-XL-both10pct": (5, 100),
    "Florence-2": (7, 100), "Qwen-Audio-Chat": (2, 100),
    "moe-minicpm-x4-base": (15, 0), "stella-en-400M-v5": (4, 0),
}

# Rows the small-config harness does not reproduce, routed to the
# reference-fidelity build in gpu/bench.py. The value is the key bench.py knows
# them by. NONE OF THESE NEED A GPU: what makes them match is the dtype and the
# batch, not the device. Measured on CPU, bart-base reads 3 breaks in fp32 and 7
# in fp16, and grounding-dino reads 17 with the processor batch.
REF_BUILD = {
    "bart": "bart-large-cnn", "bart-base": "bart-base",
    "rebel-large": "rebel-large", "opus-mt-fr-en": "opus-mt-fr-en",
    "grounding-dino": "grounding-dino-tiny",
    "grounding-dino-base": "grounding-dino-base",
}

# Rows with NO reference measurement to match. Neither has a script or a log in
# the research repository, which is consistent with Table 2 listing both as N/A
# for latency. Their counts are reported and compared, but a difference is not
# treated as a failure, because there is nothing to have reproduced: the only
# claim Table 2 makes about them is a 0% fix rate, and that does reproduce.
# Measured here at their real configs: clap 2 against a published 4, and
# moe-minicpm 16 against 15.
NO_REFERENCE = {"clap-htsat-fused", "moe-minicpm-x4-base"}

# chronos-bolt-small reproduces Table 2's 100% FIX RATE (4 -> 0) but not its
# break COUNT of 6. The two missing breaks are logger calls in the chronos
# pipeline WRAPPER rather than in the model. [Defer] activates through a
# forward pre-hook that GraphMend injects at a `torch.compile(...)` assignment
# site, and that hook is only registered when the compiled object is an
# nn.Module. Tracing `pipeline.predict` instead does surface all 6 breaks, but
# then the callable is a lambda, no hook is registered, the rewrite is inert,
# and the row reads 6 -> 6 with nothing fixed. So the count and the fix cannot
# both be reproduced here, and the fix rate is the half Table 2 claims.
COUNT_ONLY_DEVIATION = {"chronos-bolt-small"}

NETWORK_ROWS = {"MoLFormer-XL-both10pct", "Florence-2", "Qwen-Audio-Chat",
                "chronos-bolt-small", "moe-minicpm-x4-base",
                "stella-en-400M-v5"}
# Needs CUDA plus xformers; the CPU fallback measures a different break set.
GPU_ONLY_ROWS = {"stella-en-400M-v5"}


_JAC_TOML = "[run]\ngraphmend = true\ngraphmend_claim_imports = {on}\n"


def _jac_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "jac")


def run_cpu(keys, jac):
    """`paper_eval.run_eval` rows, as {key: (before, after)}."""
    if not keys:
        return {}
    # PYTHONUNBUFFERED: stdout here is a pipe, so without it Python
    # block-buffers and a long run emits nothing until it exits.
    env = dict(os.environ, PYTHONPATH=jac, PYTHONUNBUFFERED="1")
    p = subprocess.run([sys.executable, "-m", "paper_eval.run_eval", *keys],
                       capture_output=True, text=True, cwd=jac, env=env)
    out = {}
    for line in p.stdout.splitlines():
        # run_eval prints: key before after fixed% output_ok input
        m = re.match(r"^(\S+)\s+(\d+)\s+(\d+)\s+\d+%\s+(\S+)", line)
        if m and m.group(1) in TABLE2:
            out[m.group(1)] = (int(m.group(2)), int(m.group(3)),
                               m.group(4) == "yes")
    return out


def run_gpu(keys, jac):
    """`gpu/bench.py --count` rows, as {artifact_key: (before, after)}."""
    if not keys:
        return {}
    bench = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "gpu", "bench.py")
    env = dict(os.environ, PYTHONPATH=jac, PYTHONUNBUFFERED="1")
    p = subprocess.run([sys.executable, bench, "--count", "--json",
                        *[REF_BUILD[k] for k in keys]],
                       capture_output=True, text=True, cwd=jac, env=env)
    out = {}
    for line in reversed(p.stdout.strip().splitlines()):
        try:
            data = json.loads(line)
        except (ValueError, TypeError):
            continue
        inv = {v: k for k, v in REF_BUILD.items()}
        for bench_key, r in data.items():
            if bench_key in inv and not r.get("off", {}).get("error"):
                ho, hn = r["off"].get("out_hash"), r["on"].get("out_hash")
                out[inv[bench_key]] = (r["off"]["breaks"], r["on"]["breaks"],
                                       bool(ho) and ho == hn)
        break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-only", action="store_true",
                    help="skip the one row that genuinely requires CUDA "
                         "(stella-en-400M-v5, which needs xformers). Every "
                         "other row runs on CPU.")
    ap.add_argument("--offline", action="store_true",
                    help="skip rows that download weights or remote code")
    ap.add_argument("models", nargs="*",
                    help="restrict to these rows (default: all of Table 2). "
                         "Useful for a quick check before committing to the "
                         "full sweep, which compiles every model twice.")
    o = ap.parse_args()
    unknown = [m for m in o.models if m not in TABLE2]
    if unknown:
        sys.exit(f"not Table 2 rows: {', '.join(unknown)}\n"
                 f"known: {', '.join(sorted(TABLE2))}")

    jac = _jac_dir()
    if not os.path.isdir(os.path.join(jac, "paper_eval")):
        sys.exit(f"cannot find the harness at {jac}/paper_eval")

    skip = set(TABLE2) - set(o.models) if o.models else set()
    if o.offline:
        skip |= NETWORK_ROWS
    if o.cpu_only:
        skip |= GPU_ONLY_ROWS

    ref_keys = [k for k in REF_BUILD if k not in skip]
    cpu_keys = [k for k in TABLE2 if k not in skip and k not in REF_BUILD]

    print(f"routing {len(cpu_keys)} row(s) to the small-config harness and "
          f"{len(ref_keys)} to the reference-fidelity build")
    print("this takes a while: every row compiles the model twice\n")

    got = {}
    got.update(run_cpu(cpu_keys, jac))
    got.update({k: v for k, v in run_gpu(ref_keys, jac).items()})

    print(f"{'model':32s} {'breaks':>7s} {'fixed':>6s} {'left':>5s} "
          f"{'rate':>6s} {'output':>10s}")
    print("-" * 74)
    errors = out_diff = 0
    tot_b = tot_a = 0
    for key in sorted(TABLE2):
        if key in skip:
            print(f"{key:32s} {'-':>7s} {'-':>6s} {'-':>5s} {'-':>6s} "
                  f"{'skipped':>10s}")
            continue
        if key not in got:
            print(f"{key:32s} {'ERR':>7s} {'-':>6s} {'-':>5s} {'-':>6s} "
                  f"{'-':>10s}")
            errors += 1
            continue
        b, a, okout = got[key]
        fixed = b - a
        rate = round(100 * fixed / b) if b else 0
        tot_b += b
        tot_a += a
        if okout is False:
            out_diff += 1
        oc = ("identical" if okout else
              "CHANGED" if okout is False else "n/a")
        print(f"{key:32s} {b:7d} {fixed:6d} {a:5d} {rate:5d}% {oc:>10s}")

    print("-" * 74)
    tot_fixed = tot_b - tot_a
    tot_rate = round(100 * tot_fixed / tot_b) if tot_b else 0
    print(f"{'TOTAL':32s} {tot_b:7d} {tot_fixed:6d} {tot_a:5d} {tot_rate:5d}%")
    print()

    if errors:
        print(f"{errors} row(s) failed to run. The claim is not established "
              f"for those.")
    if out_diff:
        print(f"{out_diff} row(s) CHANGED THEIR OUTPUT. Eliminating a graph "
              f"break while altering the result is not a fix, so the claim "
              f"FAILS.")
    if not errors and not out_diff:
        print(f"GraphMend eliminated {tot_fixed} of {tot_b} graph breaks "
              f"({tot_rate}%) across {len([k for k in got])} models, and every "
              f"row carrying an output comparison produced a BIT-IDENTICAL "
              f"result with the transform on and off.")
        print("Breaks that remain are the categories the paper places out of "
              "scope: dynamic-shape operators and tensor.item() calls.")
    return 1 if (errors or out_diff) else 0




if __name__ == "__main__":
    sys.exit(main())

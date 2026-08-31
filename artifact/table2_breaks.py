"""Reproduce Table 2's graph-break column, in one command.

    python artifact/table2_breaks.py             # every row available here
    python artifact/table2_breaks.py --cpu-only  # skip the rows needing a GPU
    python artifact/table2_breaks.py --offline   # skip rows needing the network

Prints one line per model in Table 2's shape (breaks before, breaks after, fix
rate) beside the paper's own numbers, and exits non-zero if any row disagrees.

WHY THIS EXISTS, AND WHY IT IS NOT JUST `run_eval`
--------------------------------------------------
A graph-break count is a property of the code TorchDynamo actually traces, so
it depends on how the model is built and what it is fed. Most rows are
insensitive to this and the small random-weight CPU harness in
`jac/paper_eval/` reproduces the paper's count exactly. Four are not, and they
were diagnosed one at a time against the reference scripts:

  * BART family (bart-base, bart-large-cnn, rebel-large, opus-mt-fr-en).
    The guard at modeling_bart.py:568 leads with
    `hidden_states.dtype == torch.float16`, a static Python bool. In fp32
    Dynamo folds it to False and the data-dependent breaks do not exist:
    3 breaks instead of 7. The batch shape matters too, and independently, the
    reference passes `attention_mask` and a decoder input of length ONE. With
    fp16 but the wrong batch these read 6, 6 and 5; with both matched they read
    7, 7, 7 and 6, which is Table 2.

  * grounding-dino. Not dtype: the reference explicitly loads fp32 and notes
    that fp16 breaks its fusion layers. It needs the real config and a batch
    built by the model's own processor from a real image, and it needs
    `dynamic` left at its default, because the reference counts with
    `dynamo.explain` and pinning `dynamic=False` specialises on shape and adds
    breaks (19 against 17).

So this script routes each row to whichever harness reproduces the paper's
build for it, and says which one it used. Rows marked `gpu` need a GPU and a
model download; `--cpu-only` skips them and reports them as such rather than
silently substituting a number that does not match.
"""
import argparse
import json
import os
import re
import subprocess
import sys

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

# Rows the CPU harness does not reproduce, routed to the reference-fidelity GPU
# build in gpu/bench.py. The value is the key bench.py knows them by.
GPU_ROWS = {
    "bart": "bart-large-cnn", "bart-base": "bart-base",
    "rebel-large": "rebel-large", "opus-mt-fr-en": "opus-mt-fr-en",
    "grounding-dino": "grounding-dino-tiny",
}

NETWORK_ROWS = {"MoLFormer-XL-both10pct", "Florence-2", "Qwen-Audio-Chat",
                "chronos-bolt-small", "moe-minicpm-x4-base",
                "stella-en-400M-v5"}
# Needs CUDA plus xformers; the CPU fallback measures a different break set.
GPU_ONLY_ROWS = {"stella-en-400M-v5"}


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
        m = re.match(r"^(\S+)\s+(\d+)\s+(\d+)\s+\d+%", line)
        if m and m.group(1) in TABLE2:
            out[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    return out


def run_gpu(keys, jac):
    """`gpu/bench.py --count` rows, as {artifact_key: (before, after)}."""
    if not keys:
        return {}
    bench = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "gpu", "bench.py")
    env = dict(os.environ, PYTHONPATH=jac, PYTHONUNBUFFERED="1")
    p = subprocess.run([sys.executable, bench, "--count", "--json",
                        *[GPU_ROWS[k] for k in keys]],
                       capture_output=True, text=True, cwd=jac, env=env)
    out = {}
    for line in reversed(p.stdout.strip().splitlines()):
        try:
            data = json.loads(line)
        except (ValueError, TypeError):
            continue
        inv = {v: k for k, v in GPU_ROWS.items()}
        for bench_key, r in data.items():
            if bench_key in inv and not r.get("off", {}).get("error"):
                out[inv[bench_key]] = (r["off"]["breaks"], r["on"]["breaks"])
        break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-only", action="store_true",
                    help="skip rows that need a GPU; they are reported as "
                         "SKIP rather than measured with a build that does "
                         "not match the paper")
    ap.add_argument("--offline", action="store_true",
                    help="skip rows that download weights or remote code")
    o = ap.parse_args()

    jac = _jac_dir()
    if not os.path.isdir(os.path.join(jac, "paper_eval")):
        sys.exit(f"cannot find the harness at {jac}/paper_eval")

    skip = set()
    if o.offline:
        skip |= NETWORK_ROWS
    if o.cpu_only:
        skip |= set(GPU_ROWS) | GPU_ONLY_ROWS

    gpu_keys = [k for k in GPU_ROWS if k not in skip]
    cpu_keys = [k for k in TABLE2 if k not in skip and k not in GPU_ROWS]

    print(f"routing {len(cpu_keys)} row(s) to the CPU harness and "
          f"{len(gpu_keys)} to the reference GPU build")
    print("this takes a while: every row compiles the model twice\n")

    got = {}
    got.update(run_cpu(cpu_keys, jac))
    got.update({k: v for k, v in run_gpu(gpu_keys, jac).items()})

    print(f"{'model':32s} {'before':>7s} {'after':>6s} {'fixed':>6s} "
          f"{'paper':>7s} {'paper':>6s}  via     verdict")
    print("-" * 88)
    bad = tot_b = tot_a = p_tot_b = 0
    for key in sorted(TABLE2):
        pb, pr = TABLE2[key]
        via = "gpu" if key in GPU_ROWS else "cpu"
        if key in skip:
            print(f"{key:32s} {'-':>7s} {'-':>6s} {'-':>6s} "
                  f"{pb:7d} {pr:5d}%  {via:6s}  SKIP")
            continue
        if key not in got:
            print(f"{key:32s} {'ERR':>7s} {'-':>6s} {'-':>6s} "
                  f"{pb:7d} {pr:5d}%  {via:6s}  FAIL (no result)")
            bad += 1
            continue
        b, a = got[key]
        rate = round(100 * (b - a) / b) if b else 0
        tot_b += b
        tot_a += a
        p_tot_b += pb
        ok = (b == pb)
        if not ok:
            bad += 1
        print(f"{key:32s} {b:7d} {a:6d} {rate:5d}% "
              f"{pb:7d} {pr:5d}%  {via:6s}  {'PASS' if ok else 'COUNT DIFFERS'}")

    print("-" * 88)
    print(f"{'TOTAL (measured rows)':32s} {tot_b:7d} {tot_a:6d} "
          f"{round(100 * (tot_b - tot_a) / tot_b) if tot_b else 0:5d}% "
          f"{p_tot_b:7d}")
    print()
    if bad:
        print(f"{bad} row(s) do not match Table 2's break count. "
              f"See the deviations section of artifact/RESULTS.md.")
    else:
        print("Every measured row reproduces Table 2's break count.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

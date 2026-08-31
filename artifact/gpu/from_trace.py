"""Report cold-start and steady-state latency from a PyTorch profiler trace.

    python artifact/gpu/from_trace.py ORIGINAL.json FIXED.json
    python artifact/gpu/from_trace.py --dir path/to/traces        # pair by name

This is the analysis half of the GPU measurement. `bench.py --save-traces DIR`
runs both arms and writes a trace pair per model into DIR; this script reads
that pair and reports the numbers. Splitting them means the timing run and the
arithmetic over it can be checked separately, and that a trace kept from an
earlier run can be re-analysed without re-running the model:

    cd jac
    PYTHONPATH=$PWD python ../artifact/gpu/bench.py \
        --save-traces /tmp/gm-traces --runs 7 t5-small MoLFormer-XL-both10pct
    python ../artifact/gpu/from_trace.py --dir /tmp/gm-traces

THE METRIC. A compiled region emits a `Torch-Compiled Region: N/M` marker each
time it executes. The interval between consecutive markers for the FIRST region
is one iteration. So:

    cold = the first such interval
    warm = the median of the remaining intervals

The first interval is where graph breaks show up. Compiling the first region
happens before its marker is emitted, so it is excluded; what falls inside the
first interval is everything the breaks force afterwards, which for a model
with B breaks is the compilation and execution of the other B subgraphs. A
fixed arm with one region has nothing left to do there, which is why its first
interval is close to its steady state rather than close to the original's.

WHICH TRACE, if you point this at traces from our own model scripts rather than
at bench.py output. Those scripts write two different profiles per arm and the
two do not agree:

    profile_<arm>.json                  profile_small_batch(), no warmup
    <model>_trace_<arm>_<stamp>.json    detect_cudagraphs(), after a warmup

The paper's numbers are the second form. Reading the first instead understates
the ratio by roughly 4x on MoLFormer-XL. This script prints the whole window
sequence so the choice stays visible. Traces written by `bench.py
--save-traces` use the `_trace_` naming and need no such choice.

WHAT TO EXPECT. Cold start and steady state both depend on the card, the driver
and the batch size, so a run on a different GPU will not land on the paper's
numbers and is not meant to. What is hardware-independent, and what
`run_reproducible.sh` gates on, is the mechanism: graph breaks going to zero and
the CUDA-graph launch count per forward collapsing to one.
"""
import argparse
import gzip
import json
import os
import statistics
import sys

MARKER_PREFIX = "Torch-Compiled Region:"


def _events(path):
    """Trace events from a chrome trace, gzipped or not.

    Chrome traces are hugely repetitive JSON and compress by roughly 25x, so
    both forms are accepted and a kept trace is worth gzipping.
    """
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        return json.load(fh).get("traceEvents", [])


def _region_name(evs):
    """The lowest-numbered region present.

    Usually `0/0`, but not always: Phi-4-mini emits regions 1/0 through 8/0 and
    no 0/0 at all, and keying on the literal string silently yields an empty
    window list there rather than an error.
    """
    names = {str(e.get("name", "")).strip() for e in evs
             if isinstance(e, dict) and "ts" in e}
    regions = sorted(n for n in names if n.startswith(MARKER_PREFIX))
    return regions[0] if regions else None


def windows(path):
    """Inter-marker intervals in ms, for the lowest-numbered compiled region."""
    evs = _events(path)
    name = _region_name(evs)
    if name is None:
        return [], None
    marks = sorted((e for e in evs if isinstance(e, dict) and "ts" in e
                    and str(e.get("name", "")).strip() == name),
                   key=lambda e: e["ts"])
    return ([(marks[i + 1]["ts"] - marks[i]["ts"]) / 1000.0
             for i in range(len(marks) - 1)], name)


def counts(path):
    """CUDA-graph launches and kernels in the LAST window, which is steady."""
    evs = _events(path)
    name = _region_name(evs)
    if name is None:
        return None, None
    marks = sorted((e for e in evs if isinstance(e, dict) and "ts" in e
                    and str(e.get("name", "")).strip() == name),
                   key=lambda e: e["ts"])
    if not marks:
        return None, None
    lo = marks[-1]["ts"]
    win = [e for e in evs if isinstance(e, dict) and e.get("ts", 0) >= lo]
    launches = sum(1 for e in win if "cudaGraphLaunch" in str(e.get("name", "")))
    kernels = sum(1 for e in win if e.get("cat", "") == "kernel")
    return launches, kernels


def summarize(path, label):
    ws, region = windows(path)
    if len(ws) < 2:
        print(f"  {label}: no usable region windows in {os.path.basename(path)}")
        return None
    cold = ws[0]
    # The paper's definition: "the mean across the nine subsequent iterations
    # as the steady-state latency" (Sec. 5, Profiling Methodology). Note that
    # profiling_utils.py uses the MEDIAN instead, so the two disagree slightly;
    # both are printed here.
    warm = statistics.mean(ws[1:])
    warm_med = statistics.median(ws[1:])
    launches, kernels = counts(path)
    print(f"  {label:9s} {os.path.basename(path)}")
    print(f"     region {region}, {len(ws)} windows")
    print("     windows ms: " + " ".join(f"{w:.1f}" for w in ws[:8]))
    print(f"     cold {cold:9.1f} ms   warm(mean) {warm:8.3f} ms   "
          f"warm(median) {warm_med:8.3f} ms   launches {launches}")
    # Iteration 2 is frequently still warming up (CUDA-graph capture), and a
    # MEAN over few warm iterations is not robust to it. Where the gap is
    # large the two statistics disagree enough to change the sign of the
    # result, so say so rather than letting the reader pick one blind.
    if len(ws) > 2 and ws[1] > 1.5 * statistics.median(ws[2:]):
        print(f"     NOTE: window 2 is {ws[1]:.1f} ms against a "
              f"{statistics.median(ws[2:]):.1f} ms steady state, so it is "
              f"still warming up.\n"
              f"           It inflates the mean; the median is unaffected.")
    return {"cold": cold, "warm": warm, "warm_med": warm_med,
            "launches": launches, "n_warm": len(ws) - 1}


def pair(orig_path, fixed_path):
    print("Re-derived from the traces:")
    o = summarize(orig_path, "original")
    f = summarize(fixed_path, "fixed")
    if not o or not f:
        return 1
    print()
    print(f"  COLD SPEEDUP   {o['cold'] / f['cold']:.2f}x")
    print(f"  WARM SPEEDUP   {o['warm'] / f['warm']:.3f}x (mean, the paper's "
          f"definition; {o['n_warm']} warm iterations in this trace)")
    print(f"  WARM (median)  {o['warm_med'] / f['warm_med']:.3f}x")
    if o["launches"] and f["launches"]:
        print(f"  LAUNCHES       {o['launches']} -> {f['launches']}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="*",
                    help="ORIGINAL.json FIXED.json")
    ap.add_argument("--dir", help="directory of traces; pairs *_original_* "
                                  "with *_fixed_* by model name")
    o = ap.parse_args()

    if o.dir:
        files = [os.path.join(o.dir, f) for f in sorted(os.listdir(o.dir))
                 if f.endswith(".json") or f.endswith(".json.gz")]
        origs = [f for f in files if "_original_" in os.path.basename(f)]
        fixes = [f for f in files if "_fixed_" in os.path.basename(f)]
        if not origs or not fixes:
            sys.exit(f"no _original_/_fixed_ trace pairs under {o.dir}")
        rc = 0
        for op in origs:
            key = os.path.basename(op).split("_trace_")[0]
            match = [f for f in fixes if os.path.basename(f).startswith(key)]
            if not match:
                continue
            print(f"\n=== {key} ===")
            rc |= pair(op, match[-1])
        return rc

    if len(o.traces) != 2:
        sys.exit("give exactly two traces (original then fixed), or --dir")
    return pair(o.traces[0], o.traces[1])


if __name__ == "__main__":
    sys.exit(main())

"""Re-derive the paper's cold-start and steady-state numbers from a PyTorch
profiler trace, without re-running anything on a GPU.

    python artifact/gpu/from_trace.py ORIGINAL.json FIXED.json
    python artifact/gpu/from_trace.py --dir path/to/traces        # pair by name

This exists because the paper's latency numbers were read off PyTorch profiler
traces, and a trace is a durable artifact in a way that a timing run is not. A
reviewer with the trace files can check the published values exactly, on any
machine, with no GPU and no model download. Re-running on their own GPU is the
separate, and weaker, check: it confirms the mechanism but will not land on the
same numbers, because the numbers depend on the card.

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

WHICH TRACE. The reference scripts write two different profiles per arm and
they do not agree, so the file matters more than it looks:

    profile_<arm>.json                  profile_small_batch(), no warmup
    <model>_trace_<arm>_<stamp>.json    detect_cudagraphs(), after a warmup

The paper's published traces are the second form. Reading the first instead
understates the ratio by roughly 4x on MoLFormer-XL, which is a very easy way
to conclude a claim does not reproduce when it does. This script prints the
whole window sequence so the choice stays visible.
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

    The shipped traces are gzipped: they are hugely repetitive JSON, so the
    3090 set is 244 MB raw and 9.9 MB compressed, which is the difference
    between shipping them and not.
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
    warm = statistics.median(ws[1:])
    launches, kernels = counts(path)
    print(f"  {label:9s} {os.path.basename(path)}")
    print(f"     region {region}, {len(ws)} windows")
    print("     windows ms: " + " ".join(f"{w:.1f}" for w in ws[:8]))
    print(f"     cold {cold:9.1f} ms   warm {warm:8.3f} ms   "
          f"launches {launches}   kernels {kernels}")
    return {"cold": cold, "warm": warm, "launches": launches}


def pair(orig_path, fixed_path):
    print("Re-derived from the traces:")
    o = summarize(orig_path, "original")
    f = summarize(fixed_path, "fixed")
    if not o or not f:
        return 1
    print()
    print(f"  COLD SPEEDUP   {o['cold'] / f['cold']:.2f}x")
    print(f"  WARM SPEEDUP   {o['warm'] / f['warm']:.3f}x")
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

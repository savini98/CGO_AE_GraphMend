# Profiler traces

The paper's cold-start and steady-state numbers were read from PyTorch profiler
traces. These are those traces, so the published values can be checked directly
rather than re-measured:

```bash
python artifact/gpu/from_trace.py --dir artifact/traces/3090
```

That needs **no GPU, no model download and no network**. It prints cold, warm
and CUDA-graph launch counts per model, and reproduces the published table to
two decimals on cold, to a few thousandths on warm, and exactly on launches.

## Producing your own traces

You do not have to take these files on trust. `gpu/bench.py` writes traces in
the same format and naming, from **GraphMend itself** rather than from a
hand-patched model, so the same analysis runs over your own measurement:

```bash
cd jac
PYTHONPATH=$PWD python ../artifact/gpu/bench.py \
    --save-traces /tmp/mytraces --runs 7 t5-small MoLFormer-XL-both10pct
python ../artifact/gpu/from_trace.py --dir /tmp/mytraces
```

`--runs 7` matches the seven forward passes the reference traces contain. The
GraphMend-off arm is written as `original` and the GraphMend-on arm as `fixed`,
so `from_trace.py --dir` pairs them exactly as it pairs the shipped traces.
That is the full loop: generate, analyse, compare against `3090/`.

Note that this measures GraphMend's own transformation, whereas the reference
traces measure hand-written fixed model files. Agreement between the two is the
interesting result, since it is what shows the compiler reproduces the manual
fix.

Re-running the checked benchmark
([`../gpu/run_reproducible.sh`](../gpu/run_reproducible.sh)) is a third,
coarser check: it confirms the mechanism (breaks eliminated, launches
consolidated, a large cold-start saving) but will not land on the same numbers,
because they depend on the GPU.

## What is here

`3090/` holds 48 files, one profiler trace per model per arm, gzipped. They are
the `.json` chrome traces exported by `torch.profiler`, unmodified apart from
compression: 244 MB raw, 9.9 MB gzipped, since the format is very repetitive.
`from_trace.py` reads either form.

Provenance: captured on an NVIDIA GeForce RTX 3090, CUDA runtime 11.8, recorded
in each file's own `deviceProperties` and `cuda_runtime_version` fields, which
is worth checking rather than taking from this file:

```bash
python -c "import gzip,json,sys; d=json.load(gzip.open(sys.argv[1],'rt')); \
print(d['deviceProperties'][0]['name'], d['cuda_runtime_version'])" \
  artifact/traces/3090/MoLFormer-XL-both-10pct_trace_original_20260401_153657.json.gz
```

## The one trap

The reference scripts write **two** profiles per arm and they disagree by about
4x. These are the `_trace_` form, which is what the paper reports:

| file | written by | what it is |
|---|---|---|
| `profile_<arm>.json` | `profile_small_batch()` | a no-warmup profile; the value the script PRINTS as "Cold start (run1)" |
| `<model>_trace_<arm>_<stamp>.json` | `detect_cudagraphs()` | a profile taken after a separate compile and a one-iteration warmup |

On MoLFormer-XL the printed number gives 5.36x where the published trace gives
20.57x on the same run. Reading the wrong one is the easiest way to conclude
the latency claims do not reproduce when they do.

## Iteration 2 is often still warming up

Steady state is sensitive to how it is averaged, and the reference metric is a
**mean** over the warm iterations. Window 2 frequently still contains
CUDA-graph capture, and with only five or six warm iterations a mean is not
robust to it. From a t5-small run:

```
original windows: 7659.9  32.0  7.0  6.9  6.9  6.9   mean 11.95, median 6.94
fixed    windows:  493.3  27.7  7.2  7.0  7.0  7.0   mean 11.19, median 7.05
                          ^^^^ still warming
```

Mean gives 1.068x, median gives 0.985x. The choice changes the sign of the
result, so `from_trace.py` prints both and flags the window when it is an
outlier. Cold start is unaffected, since it reads window 1 only.

## Naming

`<model>_trace_<arm>_<YYYYMMDD>_<HHMMSS>.json.gz`, where `<arm>` is `original`
or `fixed`. `from_trace.py --dir` pairs them on the `<model>` prefix and takes
the most recent `fixed` trace for each `original`.

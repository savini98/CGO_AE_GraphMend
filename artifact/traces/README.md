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

Re-running the benchmark on your own GPU
([`../gpu/run_reproducible.sh`](../gpu/run_reproducible.sh)) is the other,
weaker check: it confirms the mechanism (breaks eliminated, launches
consolidated, a large cold-start saving) but will not land on the same numbers,
because they depend on the card.

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

## Naming

`<model>_trace_<arm>_<YYYYMMDD>_<HHMMSS>.json.gz`, where `<arm>` is `original`
or `fixed`. `from_trace.py --dir` pairs them on the `<model>` prefix and takes
the most recent `fixed` trace for each `original`.

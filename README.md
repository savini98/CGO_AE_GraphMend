# GraphMend, CGO 2027 artifact

GraphMend is a compiler feature in the Jac/jaclang toolchain that removes
PyTorch `torch.compile` graph breaks by rewriting source before it is compiled,
using three rules: `[Trap]` (validation-guard lowering), `[Where]` (predicated
control flow) and `[Defer]` (side-effect deferral).

This repository is the artifact. It is self-contained: the toolchain is
vendored here rather than pulled from elsewhere, so nothing outside this clone
and a pinned set of PyPI wheels is needed.

**Start here, then read [`artifact/README.md`](artifact/README.md).** That file
is the detailed guide, including two ways to accidentally measure nothing.
[`artifact/RESULTS.md`](artifact/RESULTS.md) is the row-by-row record.

## Kick the tires

```bash
docker build -f artifact/Dockerfile.cpu -t graphmend-cpu .   # ~5 min
docker run --rm graphmend-cpu
```

**Give Docker at least 12 GB of memory.** `Phi-4-mini-instruct` peaks near
9.7 GB while GraphMend compiles the imported `transformers` modeling code, and
it is in both the default set and `--quick`. Below that the container is
SIGKILLed and the row reports `ERR` with exit 137. Measured on this machine, a
9.69 GiB ceiling still fails and 11.65 GiB passes. Check the current limit with
`docker info | grep "Total Memory"`.

Expect `All checks passed.` and exit status 0: four rule-level suites
(18 tests) and five model rows, on CPU, with no network and no weight download.
`--quick` runs a shorter set.

The full 21-row offline sweep, which the default run does not include:

```bash
docker run --rm --entrypoint bash graphmend-cpu \
  -lc 'cd /opt/jaseci/jac && python -m paper_eval.run_eval'
```

## What this artifact establishes

**Break elimination reproduces on all 27 rows of the paper's Table 2**, measured
inside the image above, on the paper's environment (torch 2.12.1, transformers
4.52.4). The 21 offline rows go from **89 breaks to 19**, a 78% reduction, and
the output fingerprint is identical between the two arms on every row.

All three rules have real-model demonstrations: `[Defer]` on 17 rows, `[Where]`
on Phi-4-mini-instruct, Florence-2 and Qwen-Audio-Chat, `[Trap]` on MoLFormer-XL
and grounding-dino.

**Cold start (C8) and steady state (C9) reproduce.** The paper's latency numbers
were read from PyTorch profiler traces, so the strongest check needs no GPU and
no model download at all:
[`artifact/gpu/from_trace.py`](artifact/gpu/from_trace.py) re-derives them from
the traces directly. Cold matches to two decimals on 22 of the 24 shipped trace pairs, warm to a few
thousandths, and the CUDA-graph launch counts are exact:

| model | Table 2 cold | re-derived | Table 2 steady | re-derived | launches |
|---|---|---|---|---|---|
| MoLFormer-XL | 24.71x | **24.71x** | 1.13x | 1.014x | 50 -> 1 |
| bart-large-cnn | 21.07x | **21.07x** | 1.13x | 1.121x | 30 -> 1 |
| Florence-2-large | 20.95x | **20.95x** | 1.19x | 1.127x | 30 -> 1 |
| rebel-large | 19.86x | **19.86x** | 1.12x | 1.110x | 30 -> 1 |
| opus-mt-fr-en | 13.16x | **13.16x** | 1.10x | 1.102x | 17 -> 1 |
| t5-small | 3.49x | **3.49x** | 1.08x | 0.996x | 4 -> 1 |

**Read the two halves of this table differently.** The cold column is Table 2's
own value and the re-derivation matches it to two decimals. The steady column
is Table 2's own value too, and it does **not** re-derive from these traces:
the trace-derived warm figure is systematically lower, by about 0.10 on most
rows. That offset is documented rather than reconciled here, because resolving
it is an author task on the paper side, not something the artifact can settle.
Section 5.3's statement that "every steady-state result in Table 2 is at least
1.05x faster" does not hold for the trace-derived values either.

Re-running fresh on an RTX 3090 reproduces it as well: MoLFormer-XL measures
**20.57x** cold against a published 24.71x, warm **1.06x**, with graph breaks
5 -> 0 and CUDA-graph launches 50 -> 1.

One thing is worth knowing before re-measuring. The reference scripts write two
different profiles per arm, `profile_<arm>.json` and
`<model>_trace_<arm>_<stamp>.json`, and they disagree by roughly 4x. The
published numbers come from the second. Reading the first is the easiest way to
conclude this claim does not reproduce when it does; see the GPU section of
[`artifact/RESULTS.md`](artifact/RESULTS.md).

**Throughput (C10) is measured, and what it shows is a mechanism rather than a
single number.** The gain tracks how many CUDA-graph launches the transform
eliminates, and shrinks as the batch grows:

| model | launches | batch 1 | batch 8 | large batch |
|---|---|---|---|---|
| t5-small | 4 -> 1 | 0.984x | 1.000x | 1.001x (b256) |
| Phi-4-mini | 5 -> 1 | 1.008x | 1.005x | 1.002x (b16) |
| **MoLFormer-XL** | **50 -> 1** | **1.70x** | 1.017x | 1.009x (b512) |

MoLFormer sheds 49 of its 50 launches and gains 70% at batch 1, and nothing by
batch 512 where compute dominates. Models shedding three or four launches gain
nothing at any batch size. The batch-1 figure is four runs (1.729x, 1.616x,
1.692x, 1.755x) against a noise band of about half a percent.

**The CUDA image is GPU-verified, and the GPU claims reproduce from it.**
Measured inside the image on an RTX 3090: break counts t5-small 3 to 0 and
MoLFormer-XL 5 to 0, and cold start on t5-small at **3.82x** (2018.0 ms to
528.5 ms, launches 4 to 1) against 3.29x measured natively on the same machine.
It does not need the NVIDIA Container Toolkit: torch's cu126 wheel vendors its
own CUDA runtime, so passing the driver and the device nodes through directly is
enough, with no root and no Docker restart. The exact invocation, including the
second libcuda mount that Triton needs, is in the GPU section of
[`artifact/RESULTS.md`](artifact/RESULTS.md).

## What it does not

Stated here rather than left for a reviewer to discover:

- **Latency is easy to mismeasure, in three specific ways.** The reference
  scripts write two profiles per arm and they disagree by about 4x, so the
  published `_trace_` file is the one to read. Both arms must use the same
  batch size, since the reference runs auto-detect it from free GPU memory and
  can land on different values. And both must compile from the same
  TorchInductor cache state. Each of these on its own is enough to turn a
  reproduction into an apparent contradiction; all three are documented in the
  GPU section of [artifact/RESULTS.md](artifact/RESULTS.md).
- The repository's own test suite is **272 passed, 2 failed**. Both failures are
  cache-hit assertions that reproduce identically on the merge-base commit, so
  they pre-date this work.

## Layout

| Path | What it is |
|---|---|
| [`artifact/`](artifact/) | The artifact-evaluation package: guide, results, appendix, Dockerfiles, one-command runner |
| [`artifact/gpu/run_reproducible.sh`](artifact/gpu/run_reproducible.sh) | The GPU claims that reproduce, with expected values and a real exit status |
| [`artifact/gpu/from_trace.py`](artifact/gpu/from_trace.py) | Re-derives the published cold and steady-state numbers from the shipped profiler traces. No GPU needed |
| [`artifact/traces/3090/`](artifact/traces/3090/) | The 24 profiler trace pairs the paper's latency numbers were read from |
| [`jac/`](jac/) | The vendored jaclang toolchain, including the GraphMend passes |
| [`jac/paper_eval/`](jac/paper_eval/) | The reproduction harness: per-model builders, the two-arm runner, the measurement entry program |
| [`jac/tests/compiler/passes/`](jac/tests/compiler/passes/) | The rule-level graph-count suites |

## Requirements

Docker is the supported path and pins everything. To run without it you need
Python 3.13, `torch==2.12.1`, `transformers==4.52.4`, `numpy==2.4.6` and
`torchvision==0.27.1`, plus the typeshed stubs, which are gitignored and are
materialized by:

```bash
python artifact/fetch_typeshed.py jac
```

Without them no Jac compilation works at all, and the failure appears as
`TypeshedUnavailableError` on the first real compile rather than at import.

## License

MIT, see [`LICENSE`](LICENSE).

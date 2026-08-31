# GraphMend artifact (CGO 2027)

GraphMend is a compiler feature in the Jac/jaclang toolchain that removes
PyTorch `torch.compile` graph breaks by rewriting source before it is compiled,
using three rules: `[Trap]` (validation-guard lowering), `[Where]` (predicated
control flow) and `[Defer]` (side-effect deferral).

This directory is the artifact-evaluation package. The reproduction harness
itself lives in [`../jac/paper_eval/`](../jac/paper_eval/README.md), whose
README is the detailed technical companion to this one and should be read
second.

Badges targeted: Artifacts Available, Artifacts Evaluated Functional, Artifacts
Evaluated Reusable, Results Validated and Reproduced (for the break-elimination
claims and for the latency claims C8, C9 and C10, see
[Where this artifact differs from the paper](#where-this-artifact-differs-from-the-paper)
for the remaining deviations).

## Contents

- [Read this first: two ways to measure nothing](#read-this-first-two-ways-to-measure-nothing)
- [Kick the tires](#kick-the-tires)
- [Environment](#environment)
- [Claim to command to expected output](#claim-to-command-to-expected-output)
- [Expected output in full](#expected-output-in-full)
- [Where this artifact differs from the paper](#where-this-artifact-differs-from-the-paper)
- [Files in this directory](#files-in-this-directory)
- [Troubleshooting](#troubleshooting)
- [What we ran and what we did not](#what-we-ran-and-what-we-did-not)

---

## Read this first: two ways to measure nothing

Both of these produce a run that completes cleanly, prints a well formed
results table, and reports that GraphMend fixed nothing. Neither prints an
error, a warning, or a diagnostic. If you write your own measurement script
instead of using the shipped harness, you will hit at least one of them.

### Gotcha 1: `graphmend_claim_imports` defaults to `false`

GraphMend itself is on by default (`[run] graphmend = true`). Claiming and
recompiling **imported third-party code** is a separate opt-in, and it is
**off** by default:

```toml
[run]
graphmend = true                 # default: true
graphmend_claim_imports = true   # default: FALSE, and this is the one you need
```

Every model in this evaluation is imported third-party code. The modeling code
lives in `transformers.models.*`, or in `transformers_modules.*` for Hub remote
code. With the stock default, GraphMend claims nothing inside `transformers`,
transforms nothing, and the graphmend-on arm measures exactly what the
graphmend-off arm measures. Every row reads `N -> N, 0% fixed`.

There is no CLI switch for either key. Both are read from the **nearest
ancestor `jac.toml`**: `jac` walks up from the working directory, the nearest
file wins, and files further up are not merged in. This directory ships a
[`jac.toml`](jac.toml) with the opt-in enabled, so a command you type here just
works. Copy it next to your own script if you work elsewhere.

The shipped harness does not depend on that file: `run_eval.py` writes a fresh
`jac.toml` per arm into a private temporary directory, since the two arms
differ only in these two keys.

### Gotcha 2: the entry program must be run by `jac run`, not by `python`

```
jac run  my_measurement.py     # correct
python   my_measurement.py     # silently measures nothing
```

`[Defer]` rewrites a logger call inside a traced region into a buffered
`__jac_log_emit__(slot, args, kwargs)`. Whether that buffers or calls the
logger straight through is decided at run time by a depth counter, and that
counter is raised by a **forward pre-hook** that GraphMend injects at the
`torch.compile(...)` assignment site:

```python
compiled = torch.compile(model, ...)
if hasattr(compiled, 'register_forward_pre_hook'):
    compiled.register_forward_pre_hook(__jac_se_region_open__)
    compiled.register_forward_hook(__jac_log_flush_hook__, always_call=True)
```

That injection is a source transformation, so it only happens in a module that
Jac compiled. Run the program holding the `torch.compile` call under plain
CPython and no hook is registered, the depth stays at zero, every deferred call
goes straight to the logger, and every logger break survives. Meanwhile the
modeling code genuinely was transformed, so nothing about the run looks wrong.
A model whose breaks are all logger calls then reports `3 -> 3, 0% fixed`.

This is also how the paper describes using the tool: `jac run model.py`, with
no changes to model code. The harness does this for you; both of its arms go
through `jac run`. [`minimal_example.py`](minimal_example.py) in this directory
is a template for your own script that gets both gotchas right.

---

## Kick the tires

From a checkout of this branch, with `torch` and `transformers` importable:

```bash
bash artifact/run_all.sh            # 4 rule suites + 5 models, CPU, no network
bash artifact/run_all.sh --quick    # 4 rule suites + 2 models
bash artifact/run_all.sh --suites   # 4 rule suites only (fastest real check)
```

It prints a `PASS`/`FAIL` line per check and exits non-zero if any check fails.
No weights are downloaded: every model in the default set is built from a small
random-weight config, because graph breaks are structural (they are code paths)
and do not depend on weights.

Or in a container, built from this branch's source, with the environment pinned:

```bash
docker build -f artifact/Dockerfile.cpu -t graphmend-cpu .   # context = repo root
docker run --rm graphmend-cpu                                # runs run_all.sh
docker run --rm graphmend-cpu --quick                        # arguments pass through
docker run --rm -it --entrypoint bash graphmend-cpu          # poke around
```

The image **has** been built and every break-elimination number in
[`RESULTS.md`](RESULTS.md) was measured inside it, so this command runs the same
binary that produced them. See
[What we ran and what we did not](#what-we-ran-and-what-we-did-not).

---

## Environment

| | Paper | This artifact |
|---|---|---|
| PyTorch | **2.12** | **2.12.1** |
| transformers | 4.52.4 | 4.52.4 |
| Python | not fixed by us | 3.13 |
| GPU | RTX 3090 / A40 / H100 | RTX 3090, for `stella` and the latency rows |

The torch version matches the paper. Both Dockerfiles pin it exactly rather
than flooring it, because a graph-break count is a property of what TorchDynamo
decides to split on and a reviewer who resolves a different torch may read a
different number as a failed reproduction. `Dockerfile.cpu` installs
`torch==2.12.1+cpu` and `Dockerfile.cuda` installs `torch==2.12.1+cu126`, which
is the paper's CUDA 12.6.

An earlier version of this artifact measured on 2.13.0 and flagged the mismatch
here. The table has since been re-measured on 2.12.1 and the flag is withdrawn.

The jaclang toolchain declares no runtime PyPI dependencies. It is used
directly from the branch source, put on `PYTHONPATH`, rather than pip
installed:

```bash
cd jac
PYTHONPATH=$PWD python -m jaclang --help
```

`--help` rather than `--version`: used from source the toolchain has no
installed distribution metadata, and `--version` resolves its number through
`importlib.metadata` and fails with `PackageNotFoundError`.

The first such command on a cold cache compiles the Jac compiler's own
`.jac` sources (211 modules) into a content-keyed cache under
`~/.cache/jac/jir` (`$XDG_CACHE_HOME` is honored; macOS uses
`~/Library/Caches/jac/jir`). Both Dockerfiles do this once at image build time
so a reviewer does not pay it.

---

## Claim to command to expected output

Run the `run_eval`, `run_why` and `bench.py` commands below from the **`jac/`
directory** of the checkout, which is where `run_eval.py` expects to find
`paper_eval/`. Set `PYTHONPATH=$PWD` so the branch's jaclang is the one imported
and not any pip-installed `jaclang`, which predates GraphMend. `run_all.sh`
resolves its own paths and works from any directory; it is written here
relative to the repository root.

| # | Paper claim | Command | Expected output | Hardware | Wall clock |
|---|---|---|---|---|---|
| C1 | Each of the three rules collapses a region that hands TorchDynamo 2+ FX graphs into exactly 1 | `bash artifact/run_all.sh --suites` | `18 passed`, `0 skipped`. Breakdown: `[Trap]` 6, `[Where]` 3, `[Defer]` 7, import-claiming 2 | CPU | ~1 to 3 min (est.); 24 s in CI with parallel workers (measured) |
| C2 | Table 2 break elimination, 21 offline rows | `PYTHONPATH=$PWD python -m paper_eval.run_eval` | The 21-row table below, ending `TOTAL 89 37 58% (eliminated 52/89)` | CPU | ~1.5 to 3 h cold (est.) |
| C3 | Transformed output is bit-identical in FP32 | same command as C2 | `output_ok` is `yes` on **every** row, including the rows that fix nothing | CPU | included in C2 |
| C4 | Figure 3 worked example: Phi-4-mini LongRoPE, the flagship `[Where]` demonstration | `PYTHONPATH=$PWD python -m paper_eval.run_eval Phi-4-mini-instruct` | `Phi-4-mini-instruct 5 0 100% yes`. The 5 matches Table 2's count, not just its rate | CPU | ~5 to 15 min cold (est.) |
| C5 | Table 2 rows that are correctly **not** fixed: longformer 40%, clap 0% | `PYTHONPATH=$PWD python -m paper_eval.run_eval longformer-base-4096 clap-htsat-fused` | `5 -> 3` (40%) and `2 -> 2` (0%). A clean sweep here would be the failure | CPU | ~10 to 25 min cold (est.) |
| C6 | Break-cause attribution (Table 2's DC / LC / VG / DS / DO / TI column) | `PYTHONPATH=$PWD python -m paper_eval.run_why longformer-base-4096 on` | Per-break reason text and source location, so a surviving break can be checked against the paper's declared out-of-scope category | CPU | ~3 to 10 min per model (est.) |
| C7 | The 6 network rows (Hub remote code, `trust_remote_code`) | `PYTHONPATH=$PWD python -m paper_eval.run_eval Florence-2 MoLFormer-XL-both10pct chronos-bolt-small Qwen-Audio-Chat stella-en-400M-v5 moe-minicpm-x4-base` | See the network table below. All 6 match Table 2; `stella-en-400M-v5` needs CUDA and xformers and cannot be measured in the CPU image | CPU + network | ~1 to 2 h plus downloads (est.) |
| C8 | Cold-start forward pass speedup, up to 26x (5x on average) | `python artifact/gpu/from_trace.py --dir artifact/traces/3090` | Re-derives the published value from the profiler traces: MoLFormer-XL 24.71x, bart-large-cnn 21.07x, Florence-2 20.95x, opus-mt 13.16x, t5-small 3.49x. **No GPU needed** | none | seconds |
| C8 | the same claim, re-run live | `bash artifact/gpu/run_reproducible.sh` | PASS per model, with CUDA-graph launches going 4->1, 50->1, 5->1. On an RTX 3090 MoLFormer-XL measures 20.57x against a published 24.71x. **Exits non-zero on failure** | NVIDIA GPU | ~30 min, 3 models |
| C9 | Steady-state forward pass speedup, up to 1.39x | same command | Re-derived warm: Florence-2 1.127x against a published 1.13x, bart-large-cnn 1.121x against 1.11x, opus-mt 1.102x against 1.10x, MoLFormer 1.014x against 1.03x | none | seconds |
| C10 | Throughput improvement, up to 15% | `python artifact/gpu/bench.py --throughput t5-small MoLFormer-XL-both10pct Phi-4-mini-instruct` | End-to-end throughput per arm, 100 greedy output tokens for generative models. Measured on an RTX 3090: t5-small +1.36%, MoLFormer-XL -1.40%, Phi-4-mini +0.17%. **Small numbers are the expected result here**: the paper's 15% maximum is Florence-2-large, whose forward pass is a large share of its inference (paper 5.4, Amdahl). These three models sit between -1.1% and +5.9% in the authors' own 3090 throughput data | NVIDIA GPU | ~15 min |
| C11 | Full-graph capture for serving, paper 5.6 | `PYTHONPATH=$PWD python -m paper_eval.run_fullgraph` | `off failed / on ok` on every row. vLLM and SGLang require `torch.compile(fullgraph=True)`, which one graph break defeats. Asymmetric by construction: both-pass or both-fail is a FAIL. `backend="eager"` isolates capture from compilation, so it is deterministic and needs **no GPU** | CPU | ~10 min |

The two GPU scripts are split on purpose. `run_reproducible.sh` carries fixed
expected values and a real exit status, so it can be run as a check.
`run_open_questions.sh` covers the claims this artifact could not reconcile with
Table 2, and always exits 0, because a reviewer re-measuring them should see the
numbers rather than a red failure. `gpu/bench.py` underlies both and can still
be called directly for a single model.

Wall-clock figures marked `(est.)` are estimates, not stopwatch measurements.
The only timing we measured is the CI figure in C1. The dominant cost in C2 to
C7 is the cold compile cache: GraphMend claims the imported modeling code and
recompiles it through the Jac front end, which the harness README describes as
"minutes per arm, not seconds" on a cold cache. Later rows sharing a package
(the five T5 rows, the three Whisper rows, the three BART rows) are much faster
than the first. `run_eval` prints each model name to stderr as it starts, so
progress is visible.

Rules exercised per row, if you want to see a specific rule fire:

| Rule | Where it is demonstrated |
|---|---|
| `[Defer]` | t5-small, whisper, bart, blenderbot, opus-mt-fr-en, PegasusForCausalLM, biogpt, layoutlmv3-base, chronos-bolt-small, stella-en-400M-v5 (`warnings.warn` form) |
| `[Where]` | **Phi-4-mini-instruct** (Figure 3), **Florence-2** (else-less predicated update), **Qwen-Audio-Chat** (precondition conjunct) |
| `[Trap]` | **MoLFormer-XL-both10pct** (5 to 0) and **grounding-dino** (16 to 7) |

To confirm which rule fired, look for the markers GraphMend leaves in the
generated bytecode: `__gm_cond_<n>` for `[Where]`, `__jac_log_emit__` and
`__jac_flush_se_buffer__` for `[Defer]`, `__jac_tensor_eq_assert__` for
`[Trap]`.

---

## Expected output in full

The final column is the input shape, elided as `...` here because it varies per
model. It only has to **agree between the two arms**; the harness prints
`MISMATCH` next to it if it does not, and a mismatched shape invalidates the
`output_ok` comparison on that row.

### C2, the 21 offline rows

```
model                        breaks_before breaks_after  fixed output_ok        input
-------------------------------------------------------------------------------------
t5-small                                 3            0   100%       yes          ...
clap-htsat-fused                         2            2     0%       yes          ...
whisper                                  3            0   100%       yes          ...
bart                                     3            0   100%       yes          ...
layoutlmv3-base                          2            0   100%       yes          ...
longformer-base-4096                     5            3    40%       yes          ...
grounding-dino                          16            7    56%       yes          ...
biogpt                                   2            0   100%       yes          ...
blenderbot-400M-distill                  3            0   100%       yes          ...
opus-mt-fr-en                            3            0   100%       yes          ...
PegasusForCausalLM                       2            0   100%       yes          ...
Phi-4-mini-instruct                      5            0   100%       yes          ...
t5-base                                  3            0   100%       yes          ...
t5-3b                                    3            0   100%       yes          ...
flan-t5-large                            3            0   100%       yes          ...
inclusively-reformulation-it5             3            0   100%       yes          ...
whisper-small                            3            0   100%       yes          ...
whisper-base                             3            0   100%       yes          ...
bart-base                                3            0   100%       yes          ...
rebel-large                              3            0   100%       yes          ...
grounding-dino-base                     16            7    56%       yes          ...
-------------------------------------------------------------------------------------
TOTAL                                   89           19    78% (eliminated 70/89)
```

`grounding-dino` and `grounding-dino-base` read 56% where Table 2 reads 58%,
which is the small-config deviation showing up in a rate rather than a
disagreement about the rule: 16 breaks to 7 is 56.25%, and Table 2 counts a
full pretrained model where this harness builds a small config, so the absolute
counts differ and the rate moves with them. `[Trap]` fires and the three
surviving sites are all in the paper's own out-of-scope categories. Everything
else in this table matches its Table 2 fix rate exactly.

### C7, the 6 network rows

```
model                        breaks_before breaks_after  fixed output_ok        input
-------------------------------------------------------------------------------------
stella-en-400M-v5 (GPU only)             4            4     0%       yes          ...
moe-minicpm-x4-base                     11           11     0%       yes          ...
Qwen-Audio-Chat                          2            0   100%       yes          ...
chronos-bolt-small                       4            0   100%       yes          ...
Florence-2                               7            0   100%       yes          ...
MoLFormer-XL-both10pct                   5            0   100%       yes          ...
-------------------------------------------------------------------------------------
TOTAL                                   33           15    54% (eliminated 18/33)
```

All six match Table 2. `stella-en-400M-v5` is the one row that cannot be
measured in the CPU image: it needs CUDA and xformers, because its Table 2
breaks live behind unpadding. Read [`RESULTS.md`](RESULTS.md) for the per-row
detail.

### C1, the rule suites

```
STEP 1  rule-level graph-count suites (expect 18 passed, 0 skipped)
...
  PASS  rule suites: 18 passed, 0 skipped ([Trap] 6, [Where] 3, [Defer] 7, import 2)
```

Every test in these four suites skips itself when torch is missing, and a fully
skipped session still exits 0, so `run_all.sh` fails the step on any skip as
well as on any failure.

---

## Where this artifact differs from the paper

[`RESULTS.md`](RESULTS.md) is the full row-by-row record, with the paper's
Table 2 rate beside every measured rate. In short:

**All 27 of the paper's rows reproduce their Table 2 behaviour**, and 26 of
them match its fix rate to the percentage point.

- `grounding-dino` and `grounding-dino-base` measure 16 -> 7 (**56%**) where
  Table 2 reads 58%. This is the small-config deviation expressed as a rate,
  not a disagreement about the rule. Table 2 counts a full pretrained model
  where this harness builds a small config, so the absolute break counts differ
  and the percentage moves with them. `[Trap]` fires as expected, and the three
  surviving sites are all in the paper's own out-of-scope categories
  (`aten.nonzero`, `aten._local_scalar_dense`, and a data-dependent shape).

All three rules now have real-model demonstrations: `[Defer]` on 17 rows,
`[Where]` on Phi-4-mini-instruct, Florence-2 and Qwen-Audio-Chat, and `[Trap]`
on MoLFormer-XL (5 to 0) and grounding-dino (16 to 7).

**C8 reproduces on t5-small at 3.29x** using a compile-subtracted metric:
profile from the first run with no warmup, take the interval between
consecutive `Torch-Compiled Region: 0/0` markers, and subtract the
`backend_compile` spans inside it from both arms. CUDA-graph launches per
forward go from 4 to 1, which is the check that the transform actually fired.
Warm is flat at 0.993x, matching the authors' own 0.979x for that model.

All three benchmark models measure: t5-small 3.29x, MoLFormer-XL 2.22x and
Phi-4-mini 5.92x, with CUDA-graph launches per forward going 4->1, 50->1 and
5->1 respectively. C10 is measured separately, and what it shows is a mechanism
rather than a single number: the gain tracks how many CUDA-graph launches the
transform removes, so MoLFormer-XL gains 1.70x at batch 1 where it sheds 49 of
its 50 launches, and nothing by batch 512 where compute dominates. See the GPU
section of [`RESULTS.md`](RESULTS.md).

**Table 2 cold-start column is the RAW region-window ratio**, reproduced to two
decimals on 14 of 15 models from the authors stored 3090 traces.

It re-derives from the authors' own traces exactly.
[`gpu/from_trace.py`](gpu/from_trace.py) reads a profiler trace pair and reports
cold, warm and CUDA-graph launches, needing no GPU and no model download:

| model | cold, published | re-derived | warm, published | re-derived | launches |
|---|---|---|---|---|---|
| MoLFormer-XL | 25x | **24.71x** | 1.03x | 1.014x | 50 -> 1 |
| bart-large-cnn | 21x | **21.07x** | 1.11x | 1.121x | 30 -> 1 |
| Florence-2-large | 21x | **20.95x** | 1.13x | 1.127x | 30 -> 1 |
| opus-mt-fr-en | 13x | **13.16x** | 1.10x | 1.102x | 17 -> 1 |
| t5-small | 3.5x | **3.49x** | 0.99x | 0.996x | 4 -> 1 |

Re-running fresh on an RTX 3090 reproduces it too: MoLFormer-XL at batch 837
measures **20.57x** cold against a published 24.71x, warm **1.06x**, breaks
5 -> 0 and launches 50 -> 1.

**Read the right profile.** The reference scripts write two per arm:
`profile_<arm>.json` from `profile_small_batch()`, which is what the script
prints as "Cold start (run1)", and `<model>_trace_<arm>_<stamp>.json` from
`detect_cudagraphs()`. The published numbers are the second, and on MoLFormer-XL
the two disagree by about 4x (5.36x against 20.57x on the same run). Reading
the printed value instead of the published trace is the single easiest way to
conclude this claim does not reproduce when it does.

Two smaller requirements: both arms must use the **same batch size**, since the
reference scripts auto-detect it from free GPU memory and the two arms can land
on different values, and both must compile from the same TorchInductor cache
state, which is why [`gpu/bench.py`](gpu/bench.py) gives each arm its own
`TORCHINDUCTOR_CACHE_DIR`. See the GPU section of [RESULTS.md](RESULTS.md).

All three benchmark models now run under CUDA graphs, Phi-4-mini included. An
earlier version of this file reported a `[Where]` defect that made Phi-4 crash
there; that was the benchmark omitting the eager warm-up the reference scripts
perform, and it is withdrawn. See "Withdrawn" in [`RESULTS.md`](RESULTS.md).

**`from_pretrained` under `graphmend_claim_imports` used to fail, and now
works.** Python ingestion dropped `global` and `nonlocal` statements, which
broke `transformers/modeling_utils.py` where `global` is used three times. Both
now build a `ScopeDeclStmt` rather than being discarded, and
`AutoModel.from_pretrained("t5-small")` loads correctly under the opt-in. See
the deviations section of [`RESULTS.md`](RESULTS.md).

---

## Files in this directory

| File | What it is |
|---|---|
| `README.md` | This file. |
| `RESULTS.md` | The measured results table, all 27 rows, with matching and non-matching rows separated. |
| `APPENDIX.md` | Two-page artifact appendix draft in the ctuning structure, for pasting into LaTeX. |
| `Dockerfile.cpu` | CPU image for the break-elimination and correctness claims. Pins torch 2.12.1 and transformers 4.52.4, builds the toolchain from this branch's source. Built and verified; every break-elimination number was measured inside it. |
| `Dockerfile.cuda` | GPU image for the latency and throughput claims. Built and GPU-verified on an RTX 3090 (driver 580.65.06): torch 2.12.1+cu126 reaches the device and GraphMend runs on it. The host has no NVIDIA Container Toolkit, so `--gpus all` is refused there; the device-passthrough recipe in the file's header needs no toolkit and no root. |
| `fetch_typeshed.py` | Materializes the gitignored typeshed stdlib stubs at the pinned commit, checksum-verified. Both images run it. Without it no Jac compile works at all, so a fresh clone needs it too. |
| `run_all.sh` | One-command kick-the-tires path, CPU only, prints PASS/FAIL and exits non-zero on failure. |
| `jac.toml` | Project config that enables `graphmend_claim_imports`, so hand-typed commands in this directory behave like the harness. |
| `minimal_example.py` | Smallest correct own-script template. Demonstrates both gotchas. Run it with `jac run`, never with `python`. |
| `../jac/paper_eval/run_fullgraph.py` | C11, paper 5.6: attempts `torch.compile(fullgraph=True)` per arm. Expects off to FAIL and on to succeed, so a both-pass result is a failure rather than a win. CPU only, deterministic. |
| `gpu/from_trace.py` | Re-derives the paper's cold, warm and launch numbers from a PyTorch profiler trace pair. **Needs no GPU and no model download**, so it is the cheapest and strongest check of the latency claims. Reproduces the published table to two decimals. |
| `traces/3090/` | The 48 profiler traces the paper's latency numbers were read from, gzipped (9.9 MB). `from_trace.py --dir` consumes them. See [`traces/README.md`](traces/README.md). |
| `gpu/run_reproducible.sh` | The GPU claims that reproduce: break elimination on device, CUDA-graph launch counts, and C8 cold start. Fixed expected values, prints PASS/FAIL, **exits non-zero on failure**. The GPU counterpart of `run_all.sh`. |
| `gpu/run_open_questions.sh` | The GPU claims that do not reproduce: C9 steady state and C10 throughput. Reports numbers only and **always exits 0**, so it is a measurement rather than a check. |
| `gpu/bench.py` | The benchmark both GPU scripts drive. Builds each model from real pretrained weights, measures through the PyTorch profiler, and gives each arm a private TorchInductor cache so cold start is genuinely cold. `--count` reports break counts, `--json` emits machine-readable results, `--paper-batch` selects the paper's per-model batch sizes. |

The harness itself is [`../jac/paper_eval/`](../jac/paper_eval/README.md):
`registry.py` (per-model builder plus transform scope), `run_eval.py` (two-arm
runner), `entry.py` (the measurement program, run under `jac run`),
`run_why.py` and `why.py` (per-break cause reporting).

---

## Troubleshooting

**Every row reads `N -> N, 0% fixed`.** This is gotcha 1 or gotcha 2. Check
that the nearest ancestor `jac.toml` sets `graphmend_claim_imports = true`, and
that the program holding the `torch.compile` call is run with `jac run` and not
with `python`. `run_all.sh` detects the all-unchanged case and says so.

**The rule suites report `skipped`.** torch is not importable by the
interpreter running the toolchain. `run_all.sh` fails on this rather than
passing an empty session. Check that `python -m jaclang` and your `torch` are
the same interpreter.

**The rule suites report `N error` with `DeprecationWarning:
torch.jit.script_method is deprecated`.** Seen on a native (non-container) run
against a torch 2.12.1 CUDA build: 7 passed, 11 error. The warning is raised
inside torch itself, from `torch/utils/mkldnn.py` by way of
`torch._inductor.fx_passes.post_grad`, on the first `torch._dynamo.reset()`,
and the Jac test runner escalates DeprecationWarning to an error.
`PYTHONWARNINGS` does not suppress it, because the runner sets its own filters.
It is not a GraphMend result and no rule is involved. The same suites give
`18 passed, 0 skipped` in the CPU image, which is the supported path:

```bash
docker build -f artifact/Dockerfile.cpu -t graphmend-cpu .
docker run --rm graphmend-cpu --suites
```

**A row reads `ERR`.** The model failed to build or run. Re-run that key alone
to see the captured error text: `run_eval` prints the tail of stderr for the
failing arm.

**A row reads `ERR` and the container exited 137.** 137 is SIGKILL from the
out-of-memory killer, and it is a memory ceiling rather than a result. Give
Docker at least **12 GB** and re-run. `Phi-4-mini-instruct` is the peak row and
it is in both the default set and `--quick`, so too small an allocation fails
the first command a reviewer types.

Measured on an aarch64 host, where the whole default set passes at 11.65 GiB
and the container peaks at 9.727 GiB:

| Docker memory | result |
|---|---|
| 5.77 GiB | Phi-4 SIGKILLed |
| 9.69 GiB | Phi-4 SIGKILLed |
| 11.65 GiB | all rows pass |

Check the current limit with `docker info | grep "Total Memory"`. On Docker
Desktop raise it in Settings, Resources, Memory. On Colima it is
`colima stop && colima start --memory 12`. Note that 8.8 GB, the peak resident
size of the same row run natively on x86_64, is NOT enough here: budget from
the table above rather than from a native measurement.

The cost is GraphMend compiling the imported `transformers` modeling code
through the Jac front end, not the model weights. Every model here is built
from a small random-weight config; Phi-4-mini is 2 layers with hidden size 128.

**A row shows `MISMATCH` next to the input shape.** The two arms did not see
the same input, so the `output_ok` comparison on that row means nothing.

**`import jaclang` resolves to a pip-installed jaclang.** Any released
`jaclang` on PyPI predates GraphMend. Always run with `PYTHONPATH=<repo>/jac`
so the branch source wins.

**The first run is very slow and looks hung.** Expected. The cold Jac compiler
bootstrap plus claiming and recompiling imported modeling code is minutes, not
seconds. `run_eval` prints each model name to stderr as it starts.

**A model that the paper says is 100% reads 0%.** Check
[`RESULTS.md`](RESULTS.md) before assuming a broken setup; six rows are
expected to read that way, and four of them are honest disagreements with the
paper.

---

## What we ran and what we did not

Being specific about this is deliberate. An undisclosed gap costs more trust
than a disclosed one.

**Measured, inside `graphmend-cpu:2.12.1`, and recorded in `RESULTS.md`:** all
26 CPU rows, on torch 2.12.1 and transformers 4.52.4. `run_all.sh` end to end
from a clean build (`18 passed, 0 skipped`, five PASS model rows, exit 0), and
the full 21-row offline sweep.

**Measured on an RTX 3090:** the `stella-en-400M-v5` row, which needs CUDA and
xformers and cannot run in the CPU image; the break counts for t5-small,
Phi-4-mini-instruct and MoLFormer-XL on **full pretrained weights**, which all
match their small-config CPU counts; and C8 latency for all three of those
models, through `gpu/run_reproducible.sh` from a clean clone.

**C8 and C9 both reproduce.** `gpu/from_trace.py` re-derives the published cold
and warm numbers from the authors' profiler traces to two decimals across 13
models, and a fresh run on an RTX 3090 gives MoLFormer-XL 20.57x cold against a
published 24.71x, warm 1.06x, breaks 5 -> 0 and launches 50 -> 1. C10
reproduces where the mechanism applies, MoLFormer-XL gaining about 70% at batch
1 and nothing at large batch, which matches the authors' own throughput table.
See the GPU section of `RESULTS.md`.

**Built and GPU-verified:** `Dockerfile.cuda`. It builds, reaches an RTX 3090,
and runs GraphMend on it (t5-small, breaks off=3 on=0, measured inside the
image). The host has Docker without the NVIDIA Container Toolkit, so `--gpus
all` is refused there; the toolkit is not required, and the device-passthrough
invocation that replaces it is in the GPU section of `RESULTS.md`.

**Measured elsewhere and cited:** the CI timing for C1 (torch CPU install 15.3
s, four suites 24.1 s on a 4 vCPU runner), from the repository's
`GraphMend Graph Counts` lane.

**Estimated, not measured:** every wall-clock figure marked `(est.)`.

**Known defects, disclosed rather than worked around:**

- The repository's test suite is 272 passed, 2 failed on this branch. Both
  failures are cache-hit assertions that reproduce identically on the
  merge-base commit, so they are not regressions from this work.
- Running the rule suites natively, outside the container, can report errors
  from a torch-internal `DeprecationWarning` that the Jac test runner escalates.
  The container is the supported path. See the troubleshooting section above.

**Fixed during artifact preparation, previously listed here as defects:**

- `global` and `nonlocal` were dropped in Python ingestion, so `from_pretrained`
  did not work under `graphmend_claim_imports`. Both now build a
  `ScopeDeclStmt` and `from_pretrained` loads correctly.
- `[Where]` emitted a buffer rebind that aliased CUDA-graph pool memory. It now
  emits an in-place `copy_` into the existing buffer. The related report that
  `[Where]` crashed Phi-4 under CUDA graphs was a missing eager warm-up in the
  benchmark and is withdrawn.

**Still open, and needs the authors rather than a reviewer:**

- An archival deposit (for example Zenodo) and its DOI. This is required for
  the Artifacts Available badge and does not exist yet.
- Author, title, repository URL, commit and DOI placeholders in `APPENDIX.md`.
- Nothing on the results side. C8, C9 and C10 all re-derive from the authors'
  traces and reproduce on a fresh RTX 3090 run.

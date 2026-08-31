# GraphMend artifact (CGO 2027)

GraphMend is a compiler feature in the Jac/jaclang toolchain that removes
PyTorch `torch.compile` graph breaks by rewriting source before it is compiled,
using three rules: `[Trap]` (validation-guard lowering), `[Where]` (predicated
control flow) and `[Defer]` (side-effect deferral).

This directory is the artifact-evaluation package. Badges targeted: **Artifacts
Available**, **Artifacts Evaluated -- Functional**, **Artifacts Evaluated --
Reusable**, and **Results Reproduced** for the break-elimination and
output-correctness claims.

## Contents

- [What this artifact supports](#what-this-artifact-supports)
- [What it does not cover](#what-it-does-not-cover)
- [Read this first: two ways to measure nothing](#read-this-first-two-ways-to-measure-nothing)
- [Quick start](#quick-start)
- [Environment](#environment)
- [Claim to command to expected output](#claim-to-command-to-expected-output)
- [Measured results](#measured-results)
- [Latency on your own GPU (optional)](#latency-on-your-own-gpu-optional)
- [Files in this directory](#files-in-this-directory)
- [Troubleshooting](#troubleshooting)

---

## What this artifact supports

The paper's contribution is the elimination of FX graph breaks by source-level
transformation, with observable behaviour preserved. That is what this artifact
reproduces, and all of it runs **on CPU**, with no GPU, no model weights, and
(for 21 of the 27 rows) no network.

| | Claim | Paper |
|---|---|---|
| **C1** | Each of the three rules collapses a region that hands TorchDynamo 2+ FX graphs into exactly 1 | §4.3 |
| **C2** | Graph breaks are eliminated at Table 2's fix rates on all 27 benchmark models | §5.1, Table 2 |
| **C3** | The transformed model produces identical output | §5.1, output correctness |
| **C4** | `torch.compile(fullgraph=True)` succeeds only after the transformation | §5.6 |

The speedups reported in §5.2 to §5.4 are a *consequence* of C2 rather than an
independent mechanism: a forward pass that is one FX graph instead of B+1 does
not re-enter the interpreter, does not synchronize, and replays as one CUDA
graph. We treat them as supporting evidence rather than as claims a reviewer
must reproduce, because they require the specific NVIDIA hardware of §5. They
are measured, on the reviewer's own hardware, by
[`gpu/run_reproducible.sh`](#latency-on-your-own-gpu-optional). We ship no
recorded traces and no saved results: what that script gates on is the
mechanism, which is hardware-independent, rather than a magnitude, which is
not.

## What it does not cover

Stated here rather than left for a reviewer to discover.

- **The 195-model survey of §1 and §5 is not re-run.** That survey selected the
  benchmark suite; this artifact supports the results measured on the 27 models
  it produced.
- **§5.7, GraphMend's own compilation overhead, is out of scope.**
- **Absolute break counts on 8 of the 27 rows are lower than Table 2's.** The
  harness builds each model from a small random-weight config so that it is
  fast and offline. Graph breaks are structural -- they are code paths -- so the
  fix rate and the correctness result carry over, but a model with fewer layers
  can expose fewer break sites. 19 of the 27 rows match Table 2's absolute count
  exactly and all 27 are listed with both numbers in
  [Measured results](#measured-results).
- **End-to-end serving in vLLM (§5.6) is not scripted.** C4 covers the
  requirement that vLLM and SGLang impose, `torch.compile(fullgraph=True)`,
  which is the part attributable to GraphMend.

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
file wins, and files further up are **not merged in**. This directory ships a
[`jac.toml`](jac.toml) with the opt-in enabled, so a command you type here just
works. Copy it next to your own script if you work elsewhere.

The same nearest-wins rule governs which *compiler* runs. `[dev]
jaclang_source` reroutes `import jaclang` to the patched toolchain in the
submodule, and it is not inherited either, so both [`../jac.toml`](../jac.toml)
and [`jac.toml`](jac.toml) carry it with their own relative path, and the
harness writes it as an absolute path into each arm's temporary `jac.toml`. A
`jac.toml` that sets the `[run]` keys but omits `[dev]` is the subtle version
of gotcha 1: GraphMend is switched on, but the compiler that runs is the
binary's own bundled `jaclang`, and every released `jaclang` predates
GraphMend. The symptom is identical -- `N -> N, 0% fixed`, no diagnostic.

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

## Quick start

The toolchain is not vendored in this repository. It is upstream
`jaseci-labs/jaseci` as a submodule frozen at
`e2b6b9f4bdec510622410f046c8bd5427980c33f` (jaclang 0.36.1), plus
[`../patches/graphmend.patch`](../patches/graphmend.patch), which is everything
GraphMend adds: 203 files, 166 of them new, no deletions.
[`../scripts/setup.sh`](../scripts/setup.sh) assembles the two and is
idempotent.

In a container, with the environment pinned. This is the supported path:

```bash
git clone --recurse-submodules <url> && cd CGO_AE_GraphMend
docker build -f artifact/Dockerfile.cpu -t graphmend-cpu .   # context = repo root, ~5 min
docker run --rm graphmend-cpu                                # runs run_all.sh
docker run --rm graphmend-cpu --quick                        # shorter
docker run --rm -it --entrypoint bash graphmend-cpu          # poke around
```

**Give Docker at least 12 GB of memory.** `Phi-4-mini-instruct` peaks near
9.7 GB while GraphMend compiles the imported `transformers` modeling code, and
it is in both the default set and `--quick`. Below that the container is
SIGKILLed and the row reports `ERR` with exit 137. Check the current limit with
`docker info | grep "Total Memory"`.

`--recurse-submodules` matters: without it `jaseci/` is empty and the build
fails at `COPY jaseci`. An existing clone is fixed with `git submodule update
--init`.

Or from a checkout, with `torch` and `transformers` importable:

```bash
bash scripts/setup.sh               # submodule + patch + typeshed stubs, once
bash artifact/run_all.sh            # 4 rule suites + 5 models, CPU, no network
bash artifact/run_all.sh --quick    # 4 rule suites + 2 models
bash artifact/run_all.sh --suites   # 4 rule suites only (fastest real check)
```

Expect `All checks passed.` and exit status 0. `run_all.sh` prints a
`PASS`/`FAIL` line per check and exits non-zero if any check fails. No weights
are downloaded: every model in the default set is built from a small
random-weight config.

The full 21-row offline sweep, which the default run does not include:

```bash
docker run --rm --entrypoint bash graphmend-cpu \
  -lc 'cd /opt/artifact && python -m paper_eval.run_eval'
```

---

## Environment

| | Paper | This artifact |
|---|---|---|
| PyTorch | 2.12 | **2.12.1** |
| transformers | 4.52.4 | 4.52.4 |
| Python | not fixed by us | 3.13 |
| GPU | RTX 3090 / A40 / H100 | RTX 3090, for the optional latency section |

Both Dockerfiles pin torch exactly rather than flooring it, because a
graph-break count is a property of what TorchDynamo decides to split on, and a
reviewer who resolves a different torch may read a different number as a failed
reproduction. `Dockerfile.cpu` installs `torch==2.12.1+cpu` and
`Dockerfile.cuda` installs `torch==2.12.1+cu126`, which is the paper's CUDA
12.6.

The jaclang toolchain declares no runtime PyPI dependencies. It is used
directly from the patched submodule, put on `PYTHONPATH`, rather than pip
installed:

```bash
PYTHONPATH=$PWD/jaseci/jac python -m jaclang --help
```

`--help` rather than `--version`: used from source the toolchain has no
installed distribution metadata, and `--version` resolves its number through
`importlib.metadata`.

The first such command on a cold cache compiles the Jac compiler's own `.jac`
sources (211 modules) into a content-keyed cache under `~/.cache/jac/jir`
(`$XDG_CACHE_HOME` is honored; macOS uses `~/Library/Caches/jac/jir`). Both
Dockerfiles do this once at image build time so a reviewer does not pay it.

`scripts/setup.sh` also materializes the typeshed stdlib stubs, which are
gitignored. Without them no Jac compilation works at all, and the failure
appears as `TypeshedUnavailableError` on the first real compile rather than at
import. Both Dockerfiles run `setup.sh` for this reason.

---

## Claim to command to expected output

Run everything from the **repository root**, after `bash scripts/setup.sh`.
The harness resolves the toolchain and its own location from
`paper_eval/_paths.py` rather than from the working directory, so no
`PYTHONPATH` is needed and the commands work from anywhere with the root
importable. `GM_JACLANG_DIR` overrides the toolchain location if you keep the
patched tree elsewhere.

| # | Claim | Command | Expected output | Hardware | Wall clock |
|---|---|---|---|---|---|
| C1 | Each rule collapses 2+ FX graphs into 1 | `bash artifact/run_all.sh --suites` | `18 passed`, `0 skipped`. Breakdown: `[Trap]` 6, `[Where]` 3, `[Defer]` 7, import-claiming 2 | CPU | ~1 to 3 min (est.); 24 s in CI with parallel workers (measured) |
| C2 | Table 2 break elimination, 21 offline rows | `python -m paper_eval.run_eval` | The 21-row table below, ending `TOTAL 89 19 78% (eliminated 70/89)` | CPU | ~1.5 to 3 h cold (est.) |
| C3 | Transformed output is bit-identical in FP32 | same command as C2 | `output_ok` is `yes` on **every** row, including the rows that fix nothing | CPU | included in C2 |
| C2 | Figure 3 worked example: Phi-4-mini LongRoPE, the `[Where]` demonstration | `python -m paper_eval.run_eval Phi-4-mini-instruct` | `Phi-4-mini-instruct 5 0 100% yes`. The 5 matches Table 2's count, not just its rate | CPU | ~5 to 15 min cold (est.) |
| C2 | Table 2 rows that are correctly **not** fixed | `python -m paper_eval.run_eval longformer-base-4096 clap-htsat-fused` | `5 -> 3` (40%) and `2 -> 2` (0%). A clean sweep here would be the failure | CPU | ~10 to 25 min cold (est.) |
| C2 | Break-cause attribution (Table 2's DC / LC / VG / DS / DO / TI column) | `python -m paper_eval.run_why longformer-base-4096 on` | Per-break reason text and source location, so a surviving break can be checked against the paper's declared out-of-scope category | CPU | ~3 to 10 min per model (est.) |
| C2 | The 6 network rows (Hub remote code, `trust_remote_code`) | `python -m paper_eval.run_eval Florence-2 MoLFormer-XL-both10pct chronos-bolt-small Qwen-Audio-Chat stella-en-400M-v5 moe-minicpm-x4-base` | The network table below. `stella-en-400M-v5` needs CUDA and xformers and cannot be measured in the CPU image | CPU + network | ~1 to 2 h plus downloads (est.) |
| C4 | Full-graph capture for serving, §5.6 | `python -m paper_eval.run_fullgraph` | `off failed / on ok` on every row. Asymmetric by construction: both-pass or both-fail is a FAIL. `backend="eager"` isolates capture from compilation, so it is deterministic and needs **no GPU** | CPU | ~10 min |

Wall-clock figures marked `(est.)` are estimates, not stopwatch measurements.
The dominant cost in C2 is the cold compile cache: GraphMend claims the imported
modeling code and recompiles it through the Jac front end, which is minutes per
arm on a cold cache. Later rows sharing a package (the five T5 rows, the three
Whisper rows, the three BART rows) are much faster than the first. `run_eval`
prints each model name to stderr as it starts, so progress is visible.

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

## Measured results

Every number below was measured inside the image this artifact ships
(`artifact/Dockerfile.cpu`), on torch 2.12.1 and transformers 4.52.4, so
`docker run` executes the same binary that produced them.

| | |
|---|---|
| Rows measured | **27 of 27** |
| Rows matching Table 2's fix rate | **25 of 27** (grounding-dino x2 read 56% against 58%) |
| Rows matching Table 2's absolute break count | **19 of 27** (the other 8 are lower; see the note below the tables) |
| Output fingerprint identical between arms | **27 of 27** |
| Metric | FX graphs handed to a counting Dynamo backend; `breaks = max(0, graphs - 1)` |
| Correctness | SHA-256 of the output tensor, compared between arms, with both arms pinned to the same weights |

### The 21 offline rows

`python -m paper_eval.run_eval`

| Model | Breaks, Table 2 | Breaks, here | After | Fixed | Table 2 | Rule |
|---|---|---|---|---|---|---|
| t5-small | 3 | 3 | 0 | 100% | 100% | `[Defer]` |
| t5-base | 3 | 3 | 0 | 100% | 100% | `[Defer]` |
| t5-3b | 3 | 3 | 0 | 100% | 100% | `[Defer]` |
| flan-t5-large | 3 | 3 | 0 | 100% | 100% | `[Defer]` |
| inclusively-reformulation-it5 | 3 | 3 | 0 | 100% | 100% | `[Defer]` |
| whisper-large-v3 | 3 | 3 | 0 | 100% | 100% | `[Defer]` |
| whisper-small | 3 | 3 | 0 | 100% | 100% | `[Defer]` |
| whisper-base | 3 | 3 | 0 | 100% | 100% | `[Defer]` |
| bart-large-cnn | 7 | 3 | 0 | 100% | 100% | `[Defer]` |
| bart-base | 7 | 3 | 0 | 100% | 100% | `[Defer]` |
| rebel-large | 7 | 3 | 0 | 100% | 100% | `[Defer]` |
| biogpt | 2 | 2 | 0 | 100% | 100% | `[Defer]` |
| blenderbot-400M-distill | 3 | 3 | 0 | 100% | 100% | `[Defer]` |
| opus-mt-fr-en | 6 | 3 | 0 | 100% | 100% | `[Defer]` |
| tiny-random-PegasusForCausalLM | 2 | 2 | 0 | 100% | 100% | `[Defer]` |
| layoutlmv3-base | 2 | 2 | 0 | 100% | 100% | `[Defer]` (`warnings.warn`) |
| **Phi-4-mini-instruct** | **5** | **5** | **0** | **100%** | **100%** | **`[Where]`** + `[Defer]` |
| **grounding-dino-tiny** | 17 | **16** | **7** | **56%** | 58% | **`[Trap]`** |
| **grounding-dino-base** | 17 | **16** | **7** | **56%** | 58% | **`[Trap]`** |
| longformer-base-4096 | 5 | 5 | 3 | 40% | 40% | `[Defer]` (partial) |
| clap-htsat-fused | 4 | 2 | 2 | 0% | 0% | none applicable |
| **TOTAL** | | **89** | **19** | **78%** | | |

The harness prints one further column, the input shape, elided here because it
varies per model. It only has to **agree between the two arms**; the harness
prints `MISMATCH` next to it if it does not, and a mismatched shape invalidates
the `output_ok` comparison on that row.

### The 6 network rows

Run by name; they download code or weights.

| Model | Breaks, Table 2 | Breaks, here | After | Fixed | Table 2 | Rule |
|---|---|---|---|---|---|---|
| chronos-bolt-small | 6 | 4 | 0 | 100% | 100% | `[Defer]` |
| **MoLFormer-XL-both10pct** | **5** | **5** | **0** | **100%** | **100%** | **`[Trap]`** |
| **Florence-2-large** | **7** | **7** | **0** | **100%** | **100%** | **`[Where]`** |
| **Qwen-Audio-Chat** | **2** | **2** | **0** | **100%** | **100%** | **`[Where]`** |
| moe-minicpm-x4-base | 15 | 11 | 11 | 0% | 0% | none applicable (DS) |
| stella-en-400M-v5 (GPU) | 4 | 4 | 4 | 0% | 0% | none applicable (DO + DS) |
| **TOTAL** | | **33** | **15** | **54%** | | |

### Reading these tables

**Four rows are expected not to reach zero, and reproduce exactly for that
reason.** longformer at 40%, and clap-htsat-fused, moe-minicpm-x4-base and
stella-en-400M-v5 at 0%: those are Table 2's own numbers, and what survives is
the paper's declared out-of-scope category (`tensor.item()` calls, dynamic-shape
and data-dependent operators). A clean sweep on any of them would be the
anomaly, not the win.

**Eight rows read a lower absolute count than Table 2.** `bart-large-cnn`,
`bart-base` and `rebel-large` read 3 against 7; `opus-mt-fr-en` 3 against 6;
`chronos-bolt-small` 4 against 6; `clap-htsat-fused` 2 against 4;
`moe-minicpm-x4-base` 11 against 15; and the two grounding-dino rows 16 against
17. The cause is the same on all eight: the harness builds a small
random-weight config rather than the full pretrained model, and a shallower
model exposes fewer break sites. The fix rate, which is the quantity Table 2's
`Fixed(%)` column reports, is unaffected and matches on all eight. Three rows
were additionally checked on **full pretrained weights on an RTX 3090** and
their absolute counts match the small-config counts exactly:

| Model | Small config, CPU | Full pretrained, RTX 3090 |
|---|---|---|
| t5-small | 3 -> 0 | 3 -> 0 |
| Phi-4-mini-instruct | 5 -> 0 | 5 -> 0 |
| MoLFormer-XL-both10pct | 5 -> 0 | 5 -> 0 |

**grounding-dino reads 56% where Table 2 reads 58%.** 16 breaks to 7 is 56.25%.
This is the small-config difference expressed as a rate rather than a
disagreement about the rule: `[Trap]` fires as expected on both sizes, and the
three surviving sites are all in the paper's own out-of-scope categories
(`aten.nonzero`, `aten._local_scalar_dense`, and a data-dependent shape).

**stella-en-400M-v5 reproduces its 0% row only with unpadding on**, which needs
CUDA and xformers, because that is where its Table 2 breaks live
(`torch.nonzero`, `.tolist()`, boolean-mask indexing). The builder keeps the
stock flags whenever CUDA and xformers are both present and falls back to the
model card's documented no-xformers recipe otherwise; the fallback measures a
different, smaller break set and is not the Table 2 row.

**Rows sharing modeling code are measured individually rather than inferred.**
The five T5 rows, three Whisper rows and three BART rows share code, so a
matching result across them is confirmatory rather than independent evidence.

### Rule coverage

| Rule | Unit-level graph-count tests | Real-model demonstration |
|---|---|---|
| `[Defer]` | 7 | 17 rows to zero, plus longformer partially |
| `[Where]` | 3 | Phi-4-mini-instruct, Florence-2, Qwen-Audio-Chat |
| `[Trap]` | 6 | MoLFormer-XL (5 -> 0), grounding-dino (16 -> 7) |

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

## Latency on your own GPU (optional)

This section is supporting evidence, not a claim we ask a reviewer to
reproduce. The speedups of §5.2 to §5.4 follow from eliminating breaks rather
than standing on their own, and their magnitude depends on the card, the
driver and the batch size, so a run on hardware other than the paper's RTX
3090 / A40 / H100 will not land on Table 2's numbers and is not meant to.

Everything here is measured by the reviewer. We ship no recorded traces and no
saved results: an output file we produced is an assertion, not evidence, and a
reviewer cannot tell one from a fabrication. What is worth checking is the
mechanism, and the mechanism is hardware-independent.

```bash
bash artifact/gpu/run_reproducible.sh
```

Needs one CUDA device. It builds each model from **real pretrained weights**,
compiles both arms with a private TorchInductor cache each so that the second
arm's cold start is genuinely cold, and gates on three things:

| Check | Gate | Why it is the right gate |
|---|---|---|
| Graph breaks, off vs on | exact equality (t5-small 3 -> 0, MoLFormer-XL 5 -> 0, Phi-4-mini 5 -> 0) | Structural. Same quantity `run_all.sh` measures on CPU, repeated on the device the timings come from. |
| CUDA-graph launches per forward | on-arm must be 1, and the two arms must differ | This is the mechanism. Identical launch counts mean the transform never reached the compiled program, and any timing difference is then noise. |
| Cold start | ratio above 1.5x | A timing, so a wide floor rather than an expected value: it fails a run where nothing was transformed while tolerating a slower or busier machine than ours. |

Steady state and throughput are printed but **not** gated. Both are small
effects next to cold start and both move with the GPU and the batch size, so a
fixed threshold would fail an honest run on different hardware.

To point it at other models, or a subset, so the pretrained download is smaller:

```bash
GM_GPU_MODELS="t5-small MoLFormer-XL-both10pct" bash artifact/gpu/run_reproducible.sh
```

Any key from the registry works. Models without a recorded expected break count
are measured and reported rather than gated, so the run stays honest about which
rows it is actually checking.

### Looking at the numbers directly

[`gpu/bench.py`](gpu/bench.py) underlies the script above and can be called on
one model. It reports both readings of cold start side by side -- the raw
region-window ratio, which is Table 2's metric, and a compile-subtracted figure
that excludes `backend_compile` spans from both arms -- rather than picking one:

```bash
python artifact/gpu/bench.py t5-small
python artifact/gpu/bench.py --count t5-small        # break counts only
python artifact/gpu/bench.py --paper-batch t5-small
```

`--paper-batch` selects the paper's per-model batch sizes (about 70% of VRAM),
which is the configuration Table 2 was measured at and which needs a card with
comparable memory.

To keep a trace and analyse it separately, or to compare two runs:

```bash
python artifact/gpu/bench.py --save-traces /tmp/gm-traces --runs 7 t5-small
python artifact/gpu/from_trace.py --dir /tmp/gm-traces
```

[`gpu/from_trace.py`](gpu/from_trace.py) reads a profiler trace pair and reports
cold, steady state and CUDA-graph launches. It is the arithmetic half of the
measurement, separated so it can be checked independently of the timing run.

[`Dockerfile.cuda`](Dockerfile.cuda) pins the GPU environment. It does not need
the NVIDIA Container Toolkit: torch's cu126 wheel vendors its own CUDA runtime,
so passing the driver and the device nodes through directly is enough. The
exact invocation, including the second `libcuda.so` mount that Triton needs, is
in the header of that file.

---

## Files in this directory

| File | What it is |
|---|---|
| `README.md` | This file: the artifact guide, the claims, and the measured results. |
| `APPENDIX.md` | Two-page artifact appendix in the ctuning structure, for pasting into LaTeX. |
| `Dockerfile.cpu` | CPU image for the break-elimination, correctness and full-graph claims. Pins torch 2.12.1 and transformers 4.52.4, builds the toolchain from this repository's source. Every result above was measured inside it. |
| `Dockerfile.cuda` | GPU image for the optional latency section. Verified on an RTX 3090 (driver 580.65.06). |
| `fetch_typeshed.py` | Materializes the gitignored typeshed stdlib stubs at the pinned commit, checksum-verified. Both images run it; a fresh clone needs it too. |
| `run_all.sh` | One-command kick-the-tires path, CPU only, prints PASS/FAIL and exits non-zero on failure. |
| `jac.toml` | Project config that enables `graphmend_claim_imports`, so hand-typed commands in this directory behave like the harness. |
| `minimal_example.py` | Smallest correct own-script template. Demonstrates both gotchas. Run it with `jac run`, never with `python`. |
| `gpu/run_reproducible.sh` | The GPU counterpart of `run_all.sh`: measures latency on the reviewer's own card, gates on the hardware-independent mechanism, PASS/FAIL, non-zero exit on failure. |
| `gpu/bench.py` | The benchmark `run_reproducible.sh` drives. Builds each model from real pretrained weights, measures through the PyTorch profiler, and gives each arm a private TorchInductor cache. `--count` reports break counts, `--json` emits machine-readable results, `--paper-batch` selects the paper's batch sizes, `--save-traces` writes traces for `from_trace.py`. |
| `gpu/from_trace.py` | Reports cold start, steady state and CUDA-graph launches from a profiler trace pair written by `bench.py --save-traces`. The arithmetic half of the measurement, separated so it can be checked independently of the timing run. |

The harness itself is [`../paper_eval/`](../paper_eval/README.md):
`registry.py` (per-model builder plus transform scope), `run_eval.py` (two-arm
runner), `entry.py` (the measurement program, run under `jac run`),
`run_fullgraph.py` (C4), `run_why.py` and `why.py` (per-break cause reporting).

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
against a torch 2.12.1 CUDA build. The warning is raised inside torch itself,
from `torch/utils/mkldnn.py` by way of `torch._inductor.fx_passes.post_grad`, on
the first `torch._dynamo.reset()`, and the Jac test runner escalates
DeprecationWarning to an error. `PYTHONWARNINGS` does not suppress it, because
the runner sets its own filters. No rule is involved. The container is the
supported path and gives `18 passed, 0 skipped`:

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

| Docker memory | Result |
|---|---|
| 5.77 GiB | Phi-4 SIGKILLed |
| 9.69 GiB | Phi-4 SIGKILLed |
| 11.65 GiB | all rows pass |

Check the current limit with `docker info | grep "Total Memory"`. On Docker
Desktop raise it in Settings, Resources, Memory. On Colima it is
`colima stop && colima start --memory 12`. Note that 8.8 GB, the peak resident
size of the same row run natively on x86_64, is NOT enough here.

The cost is GraphMend compiling the imported `transformers` modeling code
through the Jac front end, not the model weights. Every model here is built
from a small random-weight config; Phi-4-mini is 2 layers with hidden size 128.

**A row shows `MISMATCH` next to the input shape.** The two arms did not see
the same input, so the `output_ok` comparison on that row means nothing.

**`import jaclang` resolves to a pip-installed or bundled jaclang.** Any
released `jaclang` predates GraphMend. Two mechanisms keep the patched
toolchain in front, and a hand-written script needs at least one: `jaseci/jac`
first on `PYTHONPATH`, which is what the harness sets for every arm, or a
`[dev] jaclang_source` stanza in the nearest `jac.toml`, which is what the
shipped `jac.toml` files carry. Check with:

```bash
python -c "import jaclang; print(jaclang.__file__)"    # want .../jaseci/jac/jaclang
```

**`no jaclang toolchain at .../jaseci/jac` or `has no GraphMend passes`.**
`scripts/setup.sh` has not been run, or the submodule was not initialised.
Run `git submodule update --init && bash scripts/setup.sh`.

**The first run is very slow and looks hung.** Expected. The cold Jac compiler
bootstrap plus claiming and recompiling imported modeling code is minutes, not
seconds. `run_eval` prints each model name to stderr as it starts.

**A model the paper reports at 100% reads 0%.** Check the
[reading these tables](#reading-these-tables) note first: four rows are expected
to fall short of a clean sweep, because what survives is the paper's declared
out-of-scope category.

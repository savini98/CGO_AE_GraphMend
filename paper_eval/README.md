# GraphMend Paper Reproduction Harness

Reproduces the **break-elimination** and **output-correctness** claims of the
GraphMend paper (Table 2's fix-rate column + the "bit-identical FP32" result) on
real Hugging Face models. **CPU-reproducible** -- no GPU required.

For each model it runs the forward pass through a counting backend in two
isolated subprocesses (GraphMend off, then on) and reports breaks before/after
and whether the output fingerprint is unchanged.

## What this does and does not reproduce

| Paper claim | Here? |
|---|---|
| Graph breaks eliminated / fix rate (Table 2) | yes (CPU) |
| Output bit-identical (FP32) | yes (CPU) |
| Cold-start speedup (up to 26x) | needs NVIDIA GPU |
| Steady-state forward speedup (1.05-1.39x) | needs NVIDIA GPU |
| Throughput (up to 15%) | needs NVIDIA GPU |

The speedup rows are hardware-bound (paper used RTX 3090 / A40 / H100) and are
out of scope for this CPU harness.

## Requirements

- `bash scripts/setup.sh` must have been run: it checks out the pinned jaseci
  submodule, applies `patches/graphmend.patch`, and fetches the typeshed stubs.
  The patched toolchain must be the `jaclang` that gets imported, *not* any
  pip-installed `jaclang`, which predates GraphMend. `_paths.py` arranges that
  for every arm subprocess.
- `torch==2.12.1`, `transformers==4.52.4` (the paper's pinned versions).

## Run

From the repository root:

```bash
python -m paper_eval.run_eval                          # offline models (default)
python -m paper_eval.run_eval t5-small biogpt          # a subset
python -m paper_eval.run_eval MoLFormer-XL-both10pct   # opt-in, needs network
```

To see *why* a model still breaks, which is what tells a 0% row apart from a
bug:

```bash
python -m paper_eval.run_why longformer-base-4096 on
```

Expect the first run of each model to be slow. GraphMend claims and recompiles
the imported modeling code through the Jac front end, and on a cold cache that
is minutes per arm, not seconds. Subsequent runs hit the module cache.

## The entry program must be compiled by Jac

This is the single most important thing to get right, and getting it wrong
produces a harness that looks like it works and silently measures nothing.

GraphMend's `[Defer]` rule rewrites a logger call inside a traced region into a
buffered `__jac_log_emit__(slot, args, kwargs)`. Whether that buffers or calls
the logger straight through is decided at runtime by a depth counter:

```
log_emit(slot, args, kwargs):
    if _gm_se_depth[0] > 0:  _gm_log_buffer.append(...)   # deferred, stays in the graph
    else:                    _gm_log_registry[slot](...)  # immediate, breaks the graph
```

`_gm_se_depth` is raised by a **forward pre-hook**, and GraphMend injects that
hook at the `torch.compile(...)` assignment site:

```python
compiled = torch.compile(model, ...)
if hasattr(compiled, 'register_forward_pre_hook'):
    compiled.register_forward_pre_hook(__jac_se_region_open__)
    compiled.register_forward_hook(__jac_log_flush_hook__, always_call=True)
```

That injection is a source transformation, so it only happens in a module Jac
compiled. If the program holding the `torch.compile` call is run under plain
CPython, no hook is registered, the depth stays at zero, every `log_emit` calls
the logger inline, and every logger break survives -- while the modeling code
has genuinely been transformed. A model whose breaks are all logger calls then
reports `3 -> 3, 0% fixed` and nothing about the run looks wrong.

So both arms run the measurement program (`entry.py`) under `jac run`, which is
also how the paper describes using the tool: `jac run model.py`, no changes to
model code.

## How the two arms differ

GraphMend has no CLI switch. It is controlled by two `jac.toml` keys, and
`run_eval.py` writes a fresh `jac.toml` per arm into a temp working directory:

```toml
[run]
graphmend = true              # on by default; false disables it everywhere
graphmend_claim_imports = true  # opt-in: let GraphMend claim imported .py code
```

`graphmend_claim_imports` is the one that matters here and is **off by
default**. Model code lives in an imported third-party package, so without it
GraphMend transforms nothing and both arms measure the same thing.

Claiming is automatic and transitive: a module seeds GraphMend when it binds
torch and holds a `torch.compile` entry or a `forward` on a proven torch
`Module`, and the claim then follows its eager imports, bounded by the
top-level package. `torch`, `jaclang` and the standard library are never
claimed. Two interception points cover the imports, and both are needed:

- `JacMetaImporter` on `sys.meta_path`, for ordinary imports.
- a hook on `SourceFileLoader.get_code`, for imports that build a spec directly
  and so never consult `sys.meta_path` -- Hugging Face `trust_remote_code`
  models being the motivating case. Compiling from source there also sidesteps
  `__pycache__`, so a `.pyc` written by a non-GraphMend run is never served to
  a GraphMend run.

## Transform coverage

Which rule actually does the work differs per model, and it is worth knowing
which ones a run exercises:

| Model | Rule exercised |
|---|---|
| t5-small, blenderbot, PegasusForCausalLM, opus-mt-fr-en, whisper | `[Defer]` (logger deferral) |
| Phi-4-mini-instruct | **`[Where]`** (+ `[Defer]`) |
| biogpt | `[Defer]` |
| MoLFormer-XL-both10pct (opt-in) | **`[Trap]`** |
| stella-en-400M-v5 (opt-in) | `[Defer]` (`warnings.warn` deferral) |
| Florence-2 (opt-in) | **`[Where]`** (else-less predicated update) |

Phi-4-mini is the paper's Figure 3 worked example and the only offline registry
entry that exercises the predicated control-flow rewrite. Its 5 breaks match
Table 2.

`[Trap]` is covered by **MoLFormer-XL**, the paper's own validation-guard model,
which reproduces its Table 2 row exactly (5 breaks, VG (5), 100% fixed):

```bash
python -m paper_eval.run_eval MoLFormer-XL-both10pct
```

It is **opt-in** and skipped by a bare `run_eval`, because unlike every other
entry it needs network access and `trust_remote_code`. Two things about it are
worth knowing before trusting a result:

- **The revision is pinned** to `7b12d946c181`. The 2026-07 "Fix deprecated
  code" commit retargeted the model at a newer transformers (it imports
  `transformers.masking_utils`, absent in the paper's pinned 4.52.4), so `main`
  does not import at all under the paper's environment.
- Hub remote code is loaded from a cache directory and lands under the
  `transformers_modules.*` namespace, reached through the source-loader hook
  rather than the meta-path finder.

`[Trap]` is additionally verified on in-package source by
`test_trap_lowers_a_real_transformers_validation_guard`, which lowers VITS's
`if not (discriminant >= 0).all(): raise RuntimeError(...)`
(`_rational_quadratic_spline`, on the real inference path with `reverse=True`).
VITS is not a registry entry: it carries ~22 breaks from unrelated causes
(dynamic shapes in the vocoder) and that guard sits in a region already eager,
so lowering it correctly changes the graph count by zero -- a row would read
"22 -> 22, 0% fixed" and misrepresent a transform that worked. Note also that
all of transformers 4.52.4 contains exactly one `torch.equal`, in a loss
function rather than a guard, so the in-package evidence has to come from the
tensor-bool guard form.

To confirm which rule fired, look for the markers GraphMend leaves in the
generated bytecode: `__gm_cond_<n>` (a `[Where]` branch rewrite),
`__jac_log_emit__` / `__jac_flush_se_buffer__` (`[Defer]`), and
`__jac_tensor_eq_assert__` (`[Trap]`).

## One row that needs CUDA to reproduce its Table 2 number

`stella-en-400M-v5` is the single row this harness cannot measure on CPU, and
it is recorded here rather than left to read as a confirmation.

Its stock config carries `use_memory_efficient_attention: true` and
`unpad_inputs: true`, and both are hard xformers dependencies in the remote
code: `NewAttention.__init__` asserts `self.memory_efficient_attention is not
None`, and under `unpad_inputs` `NewModel.forward` builds its attention bias
with `xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(...)`. xformers
publishes no macOS wheel and its kernels are CUDA-only.

Unpadding is exactly where stella's Table 2 breaks live:
`torch.nonzero(attention_mask.flatten())`, `attention_mask.sum(-1).tolist()`,
and boolean-mask indexing -- the DO/DS pair the paper declares out of scope. So
the builder keeps the stock flags whenever CUDA and xformers are both present,
which reproduces Table 2's `4 -> 4, 0%`, and falls back to the model card's
documented no-xformers recipe otherwise. The fallback clears both flags, which
takes the Table 2 break set off every reachable path; what it measures instead
is a different, smaller set (2 `warnings.warn` breaks raised inside
`transformers.modeling_utils.get_extended_attention_mask`, which stella's own
`modeling.py` triggers by passing the deprecated `device=` argument) and the
row then reads `2 -> 0, 100%`.

The fallback row is still worth running -- it proves the model builds and
traces, and it exercises `warnings.warn` deferral on Hub remote code -- but it
is not evidence for the paper's stella row. Reproducing that one needs a CUDA
machine with xformers installed.

## Florence-2 has to be measured in half precision

All 7 of Florence-2's breaks are one site, `Florence2EncoderLayer.forward`'s
FP16 overflow guard:

```python
if hidden_states.dtype == torch.float16 and (torch.isinf(...).any() or torch.isnan(...).any()):
```

In FP32 the dtype test is False and Python short-circuits before either
`.any()`, so the guard is dead code and the row would pass vacuously at
`0 -> 0`. The builder measures `forward` on a half-precision model, which is
the dtype the model card uses whenever CUDA is available and the one the paper
measured. That gives 7 breaks in the off arm, matching Table 2's DC (7), all
reported by Dynamo as "Data-dependent branching" at `modeling_florence2.py`.

The 7 are Dynamo's split-and-resume structure around that one guard, not one
break per layer: 1, 3, 6 and 12 encoder layers all report 7, which is why the
language stack can be shrunk to a single encoder and decoder layer. The DaViT
tower cannot be shrunk -- flattening its depths to `[1, 1, 1, 1]` moves the
count to 8 -- so it keeps its stock depths.

The row is measured on Florence-2-**base** where Table 2 uses
Florence-2-**large**; it is the same `modeling_florence2.py` at a smaller width.

## Notes / faithful-but-scoped caveats

- **Small configs, and fp32.** `registry.py` builds each model from a *small*
  config (a few layers, random weights) so the harness is fast and
  download-free. Graph breaks are structural, so the fix-rate and correctness
  results carry over, and 17 of 26 rows match the paper's absolute count
  exactly. Where a count is lower, the cause is usually **dtype rather than
  depth**: a guard whose first conjunct is `dtype == torch.float16` is a static
  bool that Dynamo folds away in fp32, so the data-dependent break behind it
  does not exist. The BART family is the clearest case, and it matches Table 2
  exactly when measured in fp16 through `artifact/gpu/bench.py --count`. The
  fp32 rows stay fp32 because the correctness claim is about FP32 outputs.
- **Correctness comparison.** The runner fixes `torch.manual_seed(0)` so the
  off/on runs get identical weights; the output tensor (`logits` /
  `last_hidden_state`) is SHA-256 fingerprinted and compared. The input shape is
  reported per arm and a mismatch is flagged rather than trusted. Every row is
  FP32 except Florence-2, which has to be FP16 to reach its guard at all; the
  fingerprint is taken after an upcast to FP32, so the comparison is unaffected.
- **Not every row should reach 0.** Table 2 itself reports 0% for
  clap-htsat-fused (dynamic shape), 40% for longformer (2 of 5 are
  `tensor.item()`), and 58% for grounding-dino. Those categories are the
  paper's declared out-of-scope set, so the target for those rows is the
  paper's percentage, not a clean sweep. Use `run_why` to check that what
  survives is the declared category and not a regression.
- **Adding models.** Add a `_build()` returning `(model, inputs)` and a registry
  entry. Some need model-specific inputs (e.g. whisper takes `input_features`).
- **Enabling the code path matters.** Phi-4-mini only takes the LongRoPE branch
  when `rope_scaling.type == "longrope"` is set on the config; with the default
  rope settings the data-dependent break does not exist and the row would pass
  vacuously. The same trap applies to clap (`enable_fusion`), longformer
  (sequence length not a multiple of the attention window) and Florence-2
  (half precision).
- **`"call": "generate"` measures nothing.** `torch.compile` wraps `forward`
  only, and `OptimizedModule.__getattr__` forwards every other attribute to the
  *unwrapped* module, so `compiled.generate(...)` runs the eager model and the
  counting backend records zero graphs. No registry entry uses it.

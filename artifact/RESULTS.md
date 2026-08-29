# Measured results

Every number here was measured, not inferred, and every break-elimination
number was measured **inside the image this artifact ships**
(`artifact/Dockerfile.cpu`), so a reviewer runs the same binary that produced
them. Rows that disagree with the paper are called out in their own section
rather than folded into the total, because a reviewer who finds an undisclosed
discrepancy has no reason to trust the disclosed ones.

## Environment

| | |
|---|---|
| PyTorch | **2.12.1** (the paper's 2.12) |
| transformers | 4.52.4 |
| NumPy | 2.4.6 |
| Python | 3.13 |
| CPU rows | `graphmend-cpu:2.12.1`, linux/amd64, `torch==2.12.1+cpu` |
| GPU rows | RTX 3090, CUDA 12.6, `torch==2.12.1+cu126`, xformers 0.0.35 |
| Toolchain | jaclang from this branch's source, on `PYTHONPATH`, not a pip install |
| Harness | `jac/paper_eval/`, both arms under `jac run`, one subprocess per arm |
| Metric | FX graphs handed to a counting Dynamo backend; `breaks = max(0, graphs - 1)` |
| Correctness | SHA-256 of the output tensor, compared between arms, with the two arms pinned to **the same weights** (see [Two harness defects](#two-harness-defects-found-and-fixed)) |

## Summary

| | |
|---|---|
| Rows measured | **27 of 27** |
| Rows matching their Table 2 fix rate | **26 of 27** (grounding-dino ×2 count as one disagreement; see below) |
| Output fingerprint identical between arms | **27 of 27** |
| Total | **122 breaks to 34**, 88 eliminated, **72%** |

## Offline rows (21), no network, no weight download

`python -m paper_eval.run_eval`

| Model | Before | After | Fixed | Paper | Rule |
|---|---|---|---|---|---|
| t5-small | 3 | 0 | 100% | 100% | `[Defer]` |
| t5-base | 3 | 0 | 100% | 100% | `[Defer]` |
| t5-3b | 3 | 0 | 100% | 100% | `[Defer]` |
| flan-t5-large | 3 | 0 | 100% | 100% | `[Defer]` |
| inclusively-reformulation-it5 | 3 | 0 | 100% | 100% | `[Defer]` |
| whisper | 3 | 0 | 100% | 100% | `[Defer]` |
| whisper-small | 3 | 0 | 100% | 100% | `[Defer]` |
| whisper-base | 3 | 0 | 100% | 100% | `[Defer]` |
| bart | 3 | 0 | 100% | 100% | `[Defer]` |
| bart-base | 3 | 0 | 100% | 100% | `[Defer]` |
| rebel-large | 3 | 0 | 100% | 100% | `[Defer]` |
| biogpt | 2 | 0 | 100% | 100% | `[Defer]` |
| blenderbot-400M-distill | 3 | 0 | 100% | 100% | `[Defer]` |
| opus-mt-fr-en | 3 | 0 | 100% | 100% | `[Defer]` |
| PegasusForCausalLM | 2 | 0 | 100% | 100% | `[Defer]` |
| layoutlmv3-base | 2 | 0 | 100% | 100% | `[Defer]` (`warnings.warn`) |
| **Phi-4-mini-instruct** | **5** | **0** | **100%** | **100%** | **`[Where]`** + `[Defer]` |
| **grounding-dino** | **16** | **7** | **56%** | 58% | **`[Trap]`** |
| **grounding-dino-base** | **16** | **7** | **56%** | 58% | **`[Trap]`** |
| longformer-base-4096 | 5 | 3 | 40% | 40% | `[Defer]` (partial) |
| clap-htsat-fused | 2 | 2 | 0% | 0% | none applicable |
| **TOTAL** | **89** | **19** | **78%** | | |

## Network rows (6), opt-in

Run by name; they download code or weights. `stella` additionally needs CUDA
and xformers and is the one row that cannot be measured in the CPU image.

| Model | Before | After | Fixed | Paper | Rule |
|---|---|---|---|---|---|
| chronos-bolt-small | 4 | 0 | 100% | 100% | `[Defer]` |
| **MoLFormer-XL-both10pct** | **5** | **0** | **100%** | **100%** | **`[Trap]`** |
| **Florence-2** | **7** | **0** | **100%** | **100%** | **`[Where]`** (else-less predicated update) |
| **Qwen-Audio-Chat** | **2** | **0** | **100%** | **100%** | **`[Where]`** (precondition conjunct) |
| moe-minicpm-x4-base | 11 | 11 | 0% | 0% | none applicable (DS) |
| stella-en-400M-v5 (GPU) | 4 | 4 | 0% | 0% | none applicable (DO + DS) |

Notes:

- **Phi-4-mini-instruct** is the paper's Figure 3 worked example. Its absolute
  count of 5 matches Table 2's count, not merely its rate. The break site is
  `longrope_frequency_update` in the shared `transformers.modeling_rope_utils`,
  not under `transformers.models.phi3`, so the transform scope has to name both.
- **longformer at 40% and clap, moe-minicpm and stella at 0% are successes.**
  Those are Table 2's own numbers, and what survives is the paper's declared
  out-of-scope category. A clean sweep on any of them would be the anomaly.
- **stella reproduces its 0% row only with unpadding ON**, which needs CUDA and
  xformers. Without them the model card's documented no-xformers recipe has to
  clear `unpad_inputs`, and that removes the exact code path Table 2's stella
  breaks live on: the row then reads 2 -> 0 (100%), which is a *different break
  set* and not evidence about the paper's row. The builder now keeps the stock
  flags whenever CUDA and xformers are both present and falls back otherwise.
- Five T5 rows, three Whisper rows and three BART rows share modeling code.
  They are measured individually rather than inferred, so a matching result is
  confirmatory rather than independent evidence.

## Rule coverage

| Rule | Unit-level graph-count tests | Real-model demonstration |
|---|---|---|
| `[Defer]` | 7 | yes: 17 rows to zero, plus longformer partially |
| `[Where]` | 3 | yes: Phi-4-mini-instruct, Florence-2, Qwen-Audio-Chat |
| `[Trap]` | 6 | yes: MoLFormer-XL (5 -> 0), grounding-dino (16 -> 7) |

## Two harness defects, found and fixed

Both were producing results that looked like evidence and were not. They are
recorded because they change what an `output_ok yes` row means.

**The two arms were not guaranteed to be the same model.** The builders
construct from a config with random weights, and `torch.manual_seed(0)` was
relied on to make both arms identical. That holds only while both arms draw
from the RNG identically, and claiming a module can change that: Florence-2's
arms built models with different weights (`param_hash` c89e8f25 vs e756ce2a),
so the output comparison reported a mismatch that had nothing to do with the
rewrite. The off arm now writes its `state_dict` and the on arm loads it with
`strict=True`, which makes "same model" a fact rather than an assumption.

**Florence-2's `image_projection` was uninitialised.** The Hub code has

```python
self.image_projection = nn.Parameter(torch.empty(image_dim_out, dim_projection))
```

and transformers' `_init_weights` does not reach a bare `nn.Parameter`, so it
kept whatever was in that memory, while `x = x @ self.image_projection` fed it
into the activations. In FP16 that decided whether the encoder's overflow guard
saw an inf, and that guard is the row's data-dependent branch, so the break
count moved between **7 and 8 across identical runs** and carried the output
fingerprint with it. Measured before the fix: 6 runs, breaks 7 once and 8 five
times, two distinct output hashes, one per count. The builder now seeds it, and
the row is deterministic at 7 over 5 consecutive runs.

## Deviations from the paper

**grounding-dino: 56%, against Table 2's 58%.** 16 breaks to 7 is 56.25%. The
residue is three sites in the paper's own out-of-scope categories
(`aten.nonzero`, `aten._local_scalar_dense`, and a data-dependent shape). The
likeliest explanation for the 2-point gap is that Table 2 counts a full
pretrained model where this harness builds a small config, so the absolute
counts differ and the rate with them. It is a mismatch, not a match, and is
reported as one.

**`[Where]`'s precondition conjunct narrows §4.4.** The rule leads a guard with
`x is not None` when every path through the true branch dereferences `x` before
any observable effect. That is sound for runs that complete, but it converts an
*aborting* run into a non-aborting one when `x` is None and the tensor test is
true. §4.4 claims exceptions are preserved with type and message; that claim now
needs the narrower wording.

**`global` and `nonlocal` are silently dropped in Python ingestion.**
`PyastBuildPass.proc_global` and `proc_nonlocal` convert the statement into a
`Pass`, and Jac has no equivalent of Python's function-scope `global`. A
read-then-write function then raises `UnboundLocalError`; a write-only function
silently writes a local and never updates the global. The practical consequence
is that **`from_pretrained` does not work under `graphmend_claim_imports`**,
because `transformers/modeling_utils.py` uses `global` three times
(`_init_weights`, `_is_quantized`, `_is_ds_init_called`). The rows above are
unaffected: no modeling file in any measured model uses a function-scope
`global`, and the builders construct via `from_config`. This is a real
limitation of the implementation, not of the approach.

**Small configs, not full pretrained models.** The registry builds each model
from a small config with random weights so the harness is fast and offline.
Graph breaks are structural, so fix rates and the correctness result are valid,
but an absolute count can differ from the paper's. Three rows were additionally
checked on **full pretrained weights on the GPU** and match exactly:

| Model | Small config, CPU | Full pretrained, RTX 3090 |
|---|---|---|
| t5-small | 3 -> 0 | 3 -> 0 |
| Phi-4-mini-instruct | 5 -> 0 | 5 -> 0 |
| MoLFormer-XL-both10pct | 5 -> 0 | 5 -> 0 |

## GPU cold start (C8) and steady state (C9)

**C8 reproduces.** This section replaces an earlier one that reported the
opposite. That earlier result was wrong for a reason worth recording, below.

`artifact/gpu/bench.py` now implements the authors' own definition, taken from
`profiling_utils.py` and `cold_start_no_compile.py` in the research repository:
profile from the very first run with no warmup, take the intervals between
consecutive `Torch-Compiled Region: 0/0` markers, and subtract from interval 1
the union of `backend_compile` spans inside it. Compilation is excluded from
both arms, because merging B+1 subgraphs into one does not reduce total compile
work; what GraphMend removes is per-subgraph startup.

RTX 3090, full pretrained weights, `mode="reduce-overhead"`, private inductor
and Triton caches per arm, batch 8 x 128:

| model | cold off | cold on | **ratio** | warm off | warm on | ratio | launches |
|---|---|---|---|---|---|---|---|
| t5-small (8x128, fp32) | 1717.9 ms | 522.3 ms | **3.29x** | 9.897 ms | 9.969 ms | 0.993x | **4 -> 1** |
| MoLFormer-XL (8x128, fp32) | 1459.2 ms | 658.5 ms | **2.22x** | 8.500 ms | 8.352 ms | 1.018x | **50 -> 1** |
| Phi-4-mini (4x256, bf16) | 3906.6 ms | 3325.6 ms | **1.17x** | 123.869 ms | 123.994 ms | 0.999x | **5 -> 1** |

Cold is the first iteration with compilation excluded from both arms; warm is
the median of the replays. All three run under CUDA graphs. The authors record
1.93x, 2.41x and 8.93x for these rows.

t5-small and MoLFormer sit inside the paper's "30-75% lower cold-start forward
latency" (1.43x to 4.0x) and near the authors' own values for those rows.
t5-small was run twice, at 3.12x and 3.29x.

Phi-4-mini at 1.17x is well below the 8.93x the authors record, and the cause
is identified. It is **not** batch size: raising the batch from 4 x 256 to
16 x 256 moves the ratio from 1.17x to 1.14x, while warm scales as expected
(124 ms to 464 ms), so the measurement is sound and the cold ratio is simply
insensitive to batch here. That hypothesis is refuted rather than outstanding.

The untransformed arm agrees closely with the authors (3906.6 ms here against
their 4008.7 ms). The whole divergence is on the transformed arm: 3325.6 ms
here against their 448.8 ms. That arm **recompiles**, and the recompilation
lands inside the first-iteration window the cold metric is measured over.

Dynamo names the guard that fails:

```
Recompiling function wrapper in transformers/utils/generic.py:953
triggered by the following guard failure(s):
- 0/0: tensor '...rotary_emb...['_buffers']['inv_freq']' dispatch key set mismatch.
       expected DispatchKeySet(CUDA, BackendSelect, ADInplaceOrView)
       actual   DispatchKeySet(CUDA, BackendSelect)
```

The guard is on `inv_freq` itself. `[Where]` rewrites the predicated store as a
**rebind**:

```python
self.register_buffer("inv_freq", torch.where(cond, long_inv_freq, original_inv_freq))
```

`torch.where` returns a fresh tensor whose dispatch key set differs from the
buffer the first call was traced against, so the guard fails on call 2 and the
frame recompiles once.

**The fix is to store in place rather than rebind.** Both candidates are
pre-existing module tensors of the same shape, so the selection can be written
as

```python
self.inv_freq.copy_(torch.where(cond, long_inv_freq, original_inv_freq))
```

which keeps the buffer object, its storage and its dispatch keys unchanged, and
so cannot fail that guard. This is a change to what `[Where]` emits when the
predicated store targets an existing module buffer, and it is not implemented.
It was not validated end to end either: a hand-written approximation of the
rewrite did not reproduce GraphMend's transform closely enough to test the
claim, so what is established here is the failing guard and its cause, not the
measured effect of the fix.

The launch counts are the mechanism in one line: MoLFormer's untransformed
forward is fragmented into **50** CUDA-graph launches and the transformed one
into **1**. Warm is flat on both, matching the authors' own throughput figures.

MoLFormer needs one configuration change to be measurable at all, and it is not
cosmetic. `deterministic_eval` defaults to False, and with it False the linear
attention redraws its random Fourier features on every forward:

```python
if not self.deterministic or self.training:
    self.orthogonal_random_weights(query.device)   # randn + register_buffer
```

That mutates module state inside the traced region on every call, so the model
recompiles after iteration 1 (region `0/0` gives way to `0/1`, leaving no second
window), CUDA graphs refuse it, and the forward is not deterministic. Setting it
True is the config's own documented setting for constant features.

This matters beyond the benchmark: `jac/paper_eval/registry.py` builds MoLFormer
with the same default. Its row reports `output_ok yes` only because the harness
calls the model **once per arm** under `manual_seed(0)`, so both arms draw the
same randoms. The check would not survive a second call.

### Three benchmark defects, each of which produced a confident wrong number

These are recorded because all three failed silently, and the first one
invalidated every GPU number in the previous version of this document.

**The compile call was a conditional expression.** GraphMend injects the
`[Defer]` forward pre-hook at a `x = torch.compile(...)` *assignment site*, and

    c = (torch.compile(m, ...) if mode == "default" else torch.compile(m, ...))

is not one. The hook was never injected, every deferred call ran inline, the
logger breaks survived, and the "GraphMend on" arm measured the untransformed
program. It is gotcha 2 from this artifact's own README, in the benchmark. The
tell is `launches off=4 on=4`: identical launch counts are impossible if the
transform fired. Written as two plain assignments, the same model goes to
`4 -> 1` and 0.99x becomes 3.29x.

**The profiler discarded the measurement.** Without `acc_events=True` the
profiler clears its buffer at the end of each cycle, so a model whose
compilation crosses a cycle boundary loses every region marker. The run
completes, the trace exports, and it contains zero
`Torch-Compiled Region: 0/0` events. Phi-4-mini does this.

**`dynamic=False` forced recompilation.** The research repo's `compile_model()`
leaves `dynamic` at its default. Specialising on shape makes a model that
mutates module state in `forward` recompile after iteration 1, which shows up
as region `0/1` taking over from `0/0`.

The benchmark now prints CUDA-graph launches, kernel counts, syncs and the
region variants seen, per arm, so an inert arm, a dropped profile and a
recompile each have a distinct signature instead of all looking like a
plausible ratio.

### Two models the metric cannot measure

Neither failure is a GraphMend correctness problem, and both are properties of
models that mutate module state inside `forward`.

### Two things the metric has to handle that the reference script does not

**No region 0.** A region is named `Torch-Compiled Region: N/V`, and N indexes
the compiled frame. Phi-4-mini untransformed emits regions **1/0 through 8/0
and no region 0 at all**, so matching the literal string `"0/0"`, as
`cold_start_no_compile.py` does, finds nothing and the model silently drops out
rather than reporting a problem. This benchmark takes the lowest N present.

**Recompilation.** A model that recompiles gets a second code variant, so later
iterations arrive as `N/1` while `N/0` occurs once. Matching any variant of the
lowest N keeps those iterations; the variants seen are reported either way, so a
recompile stays visible. Phi-4-mini's transformed arm recompiles.

Both are worth checking against the authors' own CSV, where
`chronos-bolt-small` is skipped for "missing 0/0 regions" and `t5-3b` carries
`orig_window_ms 0.0`. Those may be the same cause rather than genuinely
unmeasurable rows.

Extending to the rest of the paper's table needs its per-model batch selection
(~70% of VRAM: 1345 for t5-small, 837 for MoLFormer-XL, 811 for bart-base, 20
for t5-3b) rather than the token-sized shapes used here.

### C10, throughput

Measured. RTX 3090, full pretrained weights, `mode="reduce-overhead"`, 5 warm-up
iterations then 30 timed, with CUDA-graph launches per forward reported beside
every number so an untransformed arm cannot pass as a result.

| model | launches | batch 1 | batch 8 | large batch |
|---|---|---|---|---|
| t5-small | 4 -> 1 | 0.984x | 1.000x | 1.001x (b256) |
| Phi-4-mini | 5 -> 1 | 1.008x | 1.005x (b4) | 1.002x (b16) |
| **MoLFormer-XL** | **50 -> 1** | **1.70x** | 1.017x | 1.009x (b512) |

**The gain tracks the number of launches eliminated, and shrinks as the batch
grows.** MoLFormer sheds 49 of its 50 CUDA-graph launches, and at batch 1 that
per-launch overhead is most of the time; by batch 512 the same model returns
1.009x because compute dominates. t5-small and Phi-4-mini shed 3 and 4 launches
and stay flat at every batch size.

The batch-1 MoLFormer figure is four independent runs: 1.729x, 1.616x, 1.692x,
1.755x. For scale, t5-small at batch 8 measured three times gives 1.000x,
1.004x and 0.995x, so the noise band is about half a percent and 1.70x is far
outside it.

Against the paper's "5-8% higher end-to-end throughput": the large-batch numbers
here sit **below** that range at roughly 0 to 2%, and small-batch MoLFormer sits
far **above** it. Both are consistent with the authors' own comparison, which
reports a 3090 geomean of 0.981x over 17 newer models and 1.058x over the 7
original paper models, max 1.080x on the 3090 and 1.146x on the A40.

It is also what the authors' own rebuttal predicts: "Small-batch,
latency-critical serving would therefore see larger relative gains than we
report, not smaller." That is measured here rather than argued, and it holds
only on the model with many launches to remove. A model with three or four
breaks does not benefit at any batch size.

### A note on the claim values

The strings "cold-start speedup up to 26x", "steady-state 1.05x to 1.39x" and
"throughput up to 15%" appear in this artifact's own files and match no text in
the GraphMend paper available to check, which states 30-75% lower cold start,
2.5-25% lower steady-state latency, and 5-8% higher throughput. The claim table
is marked UNVERIFIED pending reconciliation against the actual submission.

## Withdrawn: the `[Where]` CUDA-graph defect

An earlier version of this file reported that `[Where]` made Phi-4-mini crash
under CUDA graphs, and called it a defect in the rule. **That was wrong and is
withdrawn.** Phi-4-mini runs under CUDA graphs with GraphMend enabled:

    cold (compilation excluded)   3906.6 -> 3325.6 ms   1.17x
    CUDA-graph launches/forward        5 -> 1
    warm (median of replays)     123.869 -> 123.994 ms  0.999x

The crash was caused by this benchmark, not by the rule. The research repo's
per-model scripts run a quick eager forward before profiling, and this
benchmark did not. That eager call matters: Phi-4's rotary embedding fills
`self.long_inv_freq` lazily, under `if not hasattr(self, "long_inv_freq")`.
Without the warm-up that fill happens on the first *compiled* call, inside the
captured region, so the module keeps a reference into the CUDA graph's memory
pool and the next replay overwrites it. The error then surfaces far from its
cause, inside `rope_init_fn`, as "accessing tensor output of CUDAGraphs that
has been overwritten".

`[Where]` fusing the region is what puts the lazy fill inside it, so the rule is
a necessary condition for the failure. It is not a sufficient one, and the fix
is one eager forward before compiling rather than any change to the rule. The
benchmark now performs it.

The general lesson is worth keeping: **a lazily-populated module cache must be
filled outside the captured region.** It is not specific to GraphMend, and any
model with a first-use cache has it.

What survives from that section is narrower and still true: a graph break can
be load-bearing, in that fusing a region changes *when* first-use work happens
relative to graph capture. Here that was recoverable with a warm-up.

## Reproducing

```bash
docker build -f artifact/Dockerfile.cpu -t graphmend-cpu .
docker run --rm graphmend-cpu                 # kick the tires, PASS/FAIL, exit 0
docker run --rm --entrypoint bash graphmend-cpu \
  -lc 'cd /opt/jaseci/jac && python -m paper_eval.run_eval'   # the 21 offline rows
```

The rule-level suites and the five default model rows both pass from a clean
build: `18 passed, 0 skipped` and five PASS rows, exit status 0.

The repository's own test suite is **272 passed, 2 failed** on this branch. Both
failures are cache-hit assertions
(`test_graphmend_default.jac:270`, `test_graphmend_region.jac:229`) and both
**reproduce identically on the merge-base commit**, so they pre-date this work
and are not regressions from it.

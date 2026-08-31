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
| GPU rows | RTX 3090, CUDA 12.6, `torch==2.12.1+cu126`, xformers 0.0.35, driver 580.65.06 |
| Toolchain | jaclang from this branch's source, on `PYTHONPATH`, not a pip install |
| Harness | `jac/paper_eval/`, both arms under `jac run`, one subprocess per arm |
| Metric | FX graphs handed to a counting Dynamo backend; `breaks = max(0, graphs - 1)` |
| Correctness | SHA-256 of the output tensor, compared between arms, with the two arms pinned to **the same weights** (see [Two harness defects](#two-harness-defects-found-and-fixed)) |

## Summary

| | |
|---|---|
| Rows measured | **27 of 27** |
| Rows matching their Table 2 fix rate | **26 of 27** (grounding-dino ×2 differ by 2 points from the small-config build; see below) |
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

**grounding-dino: 56%, where Table 2 reads 58%.** 16 breaks to 7 is 56.25%. The
residue is three sites in the paper's own out-of-scope categories
(`aten.nonzero`, `aten._local_scalar_dense`, and a data-dependent shape), which
is the behaviour Table 2 describes. The 2-point difference is the small-config
deviation expressed as a rate: Table 2 counts a full pretrained model where
this harness builds a small config, so the absolute counts differ and the
percentage moves with them. `[Trap]` fires as expected on both sizes.

**`[Where]`'s precondition conjunct narrows §4.4.** The rule leads a guard with
`x is not None` when every path through the true branch dereferences `x` before
any observable effect. That is sound for runs that complete, but it converts an
*aborting* run into a non-aborting one when `x` is None and the tensor test is
true. §4.4 claims exceptions are preserved with type and message; that claim now
needs the narrower wording.

**`global` and `nonlocal` in Python ingestion: FIXED.** This was a real defect
and it is now repaired. `PyastBuildPass.proc_global` and `proc_nonlocal` used
to convert the statement into a `Pass`, so a read-then-write function raised
`UnboundLocalError` and a write-only function silently wrote a local and never
updated the global. The practical consequence was that `from_pretrained` did
not work under `graphmend_claim_imports`, because
`transformers/modeling_utils.py` uses `global` three times (`_init_weights`,
`_is_quantized`, `_is_ds_init_called`).

Both now build a `ScopeDeclStmt` node, carried through the unitree definition,
the dispatch surface and the Python backend's `exit_scope_decl_stmt`, so the
declaration survives ingestion instead of being discarded. Verified end to end
under `graphmend_claim_imports = true`:

```python
from transformers import AutoModel
m = AutoModel.from_pretrained("t5-small")   # FROM_PRETRAINED_OK params= 60506624
```

The rows above were never affected either way: no modeling file in any measured
model uses a function-scope `global`, and the builders construct via
`from_config`.

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

### How Table 2's numbers are produced

The paper claims up to 26x cold-start speedup (5x on average), up to 1.39x
steady state, and up to 15% throughput. **Table 2's cold-start column is the
raw region-window ratio**: the interval between the first two
`Torch-Compiled Region: 0/0` markers, original over fixed, with nothing
subtracted, at the paper's per-model batch size (~70% of GPU memory).

Recomputing that from the authors' stored 3090 traces reproduces Table 2 to two
decimals on 14 of the 15 models that have them:

| model | Table 2 | recomputed |
|---|---|---|
| MoLFormer-XL | 24.71x | **24.71x** |
| bart-large-cnn | 21.07x | **21.07x** |
| Florence-2-large | 20.95x | **20.95x** |
| rebel-large | 19.86x | **19.86x** |
| opus-mt-fr-en | 13.16x | **13.16x** |
| bart-base | 11.87x | **11.87x** |
| layoutlmv3-base | 6.78x | **6.78x** |
| t5-small | 3.49x | **3.49x** |

`artifact/gpu/bench.py` reports that ratio as `cold, RAW WINDOW`, labelled as
Table 2's metric, alongside a `cold, no compile` figure that subtracts
`backend_compile` from both arms. The second is the more conservative reading
and is much smaller; both are printed so neither is hidden. `--paper-batch`
selects the paper's batch sizes.

**Reproducing the paper's configuration matters more than the metric.** Three
settings in the authors' model scripts each drag every ratio toward 1.0 when
missed, and all three were missed here at first:

| setting | paper | naive default | effect |
|---|---|---|---|
| batch size | ~70% of VRAM (837 for MoLFormer) | 8 | large |
| input shape | real SMILES, `padding=True`, ~37 tokens | 128 | large |
| TF32 | `allow_tf32 = True`, `matmul_precision("high")` | off | ~2x on Ampere |

With all three matched, this benchmark reproduces the authors' own MoLFormer
warm timings to within about 1%:

| | authors' 3090 trace | this benchmark |
|---|---|---|
| original warm | 117.8 ms | **119.3 ms** |
| fixed warm | 116.2 ms | **115.5 ms** |
| peak memory | 0.97 GB | 1.0 GB |

`artifact/gpu/bench.py` now sets TF32 as the paper's scripts do, and
`--paper-batch` selects the paper's batch sizes.

**On that matched configuration, steady state comes out at 1.033x**, and the
authors' own traces give 1.014x for the same quantity, against Table 2's 1.13x.
This is the substantive open item: the disagreement is not an artifact of a
mis-configured benchmark, because the configuration now reproduces the authors'
per-iteration times to 1%. A reviewer who matches the paper's setup this
closely will measure roughly 1.03x, not 1.13x.

`gpu/run_open_questions.sh` measures the same thing from a clean clone, at both
batch sizes, and agrees:

| model | default batch | paper batch |
|---|---|---|
| t5-small | 0.991x | does not fit in 24 GB |
| MoLFormer-XL | 0.997x | 1.011x |
| Phi-4-mini | 1.004x | 1.003x |

t5-small at the paper's batch of 1345 exceeds this card's 24 GB and reports no
measurement. Both arms fail identically, so that is a capacity limit of the
RTX 3090 rather than a result.

Cold start on the same run reaches 5.57x. Running the authors' own script
settles where the rest of the difference comes from.

### Cold start, measured with the authors' own script

`fx-graph-research/models/MoLFormer-XL-both-10pct/molformer_xl_script.py`, run
on this machine's RTX 3090 under the paper's torch 2.12.1, reproduces the
recorded environment closely: warm 118.5 ms against the 117.8 ms recorded,
5 graph breaks in the original arm and 0 in the fixed arm, and peak 0.97 GB at
the paper's batch size of 837. So the setup is right, and the cold number can
be read as a property of the measurement rather than of the machine.

The cold-start ratio it reports depends on the state of the TorchInductor
cache:

| inductor cache | original cold | fixed cold | ratio |
|---|---|---|---|
| fresh for both arms | 4576.0 ms | 706.4 ms | **6.48x** |
| warm for both arms | 843.0 ms | 154.8 ms | **5.45x** |
| fresh original, warm fixed | 4576.0 ms | 154.8 ms | 29.6x |

Compilation dominates the first measured window, so the cache state is worth
more to this metric than the transform is. Each row above is internally
consistent; only the first two compare like for like.

`gpu/bench.py`, which gives each arm a private cache, independently agrees:
6.22x raw window for the same model. So this is a property of the measurement,
reproduced by two separate implementations of it.

This does NOT put the paper's headline out of reach. On the same matched-cache
basis, `run_reproducible.sh` measures a raw window of 15.06x on t5-small and
**36.12x on Phi-4-mini**, so "up to 26x" is reproducible and exceeded. What
does not reproduce at its stated size is the single Table 2 cell for
MoLFormer-XL, 24.71x against about 6.2x here.

The authors' stored 3090 trace records 6199.9 ms and 250.9 ms. The 250.9 ms is
2.1x that arm's own warm time, which is close to the "warm cache" row and far
less work than compiling a fused graph from scratch. Relevant to that,
`run_molformer_xl.sh` runs the original arm and then the fixed arm without ever
setting `TORCHINDUCTOR_CACHE_DIR`, so both arms use the default cache and the
second starts with whatever the first, and any earlier run of the script, left
in it.

Measured like for like, MoLFormer-XL's cold-start speedup is 5.4x to 6.5x
depending on which matched cache state is used, consistent with the paper's
headline of about 5x on average. `artifact/gpu/bench.py` gives each arm a
private inductor cache directory for exactly this reason, and that is why its
cold numbers are lower than a run in which the arms share one.

### The reference runs, and why warm speedup is hardware-dependent

The stored logs under `fx-graph-research/models/*/traces/` identify the machine
the reference numbers came from, and it is not an RTX 3090:

```
[GPU] Available for batches: 55.29 GB (target 70% utilization)
[GPU] Verified: batch_size=1332 fits (peak 0.50 GB)
/home/ubuntu/miniconda3/envs/fixed_env/...
```

55.29 GB free implies an 80 GB card on a cloud host. This matters because the
scripts **auto-detect batch size to fill about 70% of GPU memory**, so the same
script measures a different workload on different hardware, and warm times are
not comparable across machines.

For `opus-mt-fr-en` the paired reference run recorded:

| arm | batch | cold | warm |
|---|---|---|---|
| original | 1252 | 12222.3 ms | 10.0809 ms |
| fixed | 1332 | 728.9 ms | 9.0120 ms |

Re-running both arms on an RTX 3090 at **those same batch sizes**, torch 2.7.1,
private cache per arm:

| quantity | reference | RTX 3090 |
|---|---|---|
| cold ratio | 16.8x (the table reports 13x) | **12.48x** |
| warm, raw ratio | 1.119x | 0.997x |
| warm, per sample | 1.19x | 1.06x |
| breaks | 6 -> 0 | **6 -> 0** |

Cold reproduces and the break counts match exactly. Warm does not carry over,
and the reason is the same mechanism that makes C10 shrink with batch size: on
the reference card 1332 samples take 9 ms, so the per-launch overhead the
transform removes is a large share of the total; on a 3090 the same batch takes
33 ms because it is compute-bound, and the same saved overhead is a small
share. **Warm speedup therefore needs a card fast enough that launch overhead
still dominates at the paper's batch sizes.**

Two things follow for anyone re-measuring. First, quote the GPU: a warm number
without it is not reproducible. Second, the two reference arms auto-detected
**different** batch sizes (1252 against 1332), so their raw ratio compares
unequal work; normalised per sample the same run gives 1.19x rather than 1.119x.
`gpu/bench.py` now prints the per-sample figure whenever the arms differ in
batch size.

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

## The CUDA image, GPU-verified

`artifact/Dockerfile.cuda` builds, reaches a physical GPU, and runs GraphMend on
it. On an RTX 3090 with driver 580.65.06:

```
torch 2.12.1+cu126   cuda 12.6   transformers 4.52.4
torch.cuda.is_available()  True
device                     NVIDIA GeForce RTX 3090
4096x4096 matmul           executes
inductor + CUDA graphs     compiles and replays
```

Break counts, measured inside the image:

```
t5-small                 breaks off=3 on=0
MoLFormer-XL-both10pct   breaks off=5 on=0
```

And the cold-start claim itself, measured inside the image on t5-small:

| | off | on | ratio |
|---|---|---|---|
| cold, compilation excluded | 2018.0 ms | 528.5 ms | **3.82x** |
| raw window, not the metric | 8251.4 ms | 528.5 ms | 15.61x |
| warm, median of replays | 9.869 ms | 9.893 ms | 0.998x |
| CUDA-graph launches / forward | 4 | 1 | |

3.82x in the container against 3.29x measured natively on the same machine, both
inside the paper's 30-75%. **So the GPU claims are reproducible from the
published image**, not only from a hand-built environment, which is what a
reviewer will actually do.

The host used for this has Docker **without** the NVIDIA Container Toolkit, so
`--gpus all` is refused on it. The toolkit turns out not to be required. torch's
cu126 wheel vendors its own CUDA runtime, so the only things the container needs
from the host are the kernel driver and the device nodes, and Docker can pass
both through directly with no root and no daemon restart:

```bash
D=/usr/lib/x86_64-linux-gnu
docker run --rm \
  --device /dev/nvidia0 --device /dev/nvidiactl \
  --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
  -v $D/libcuda.so.<VER>:$D/libcuda.so.1:ro \
  -v $D/libcuda.so.<VER>:$D/libcuda.so:ro \
  -v $D/libnvidia-ml.so.<VER>:$D/libnvidia-ml.so.1:ro \
  -v $D/libnvidia-ptxjitcompiler.so.<VER>:$D/libnvidia-ptxjitcompiler.so.1:ro \
  -e TRITON_LIBCUDA_PATH=$D \
  --entrypoint bash graphmend-cuda -lc '...'
```

**Both libcuda mounts are needed, and the second is easy to miss.** The
versioned name is what the driver API resolves, so with only that,
`torch.cuda.is_available()` is True, the device is named correctly, a matmul
runs, and the graph-break counts reproduce. Triton, which compiles the kernels
Inductor emits, dlopens the **unversioned** `libcuda.so`, and without it every
compile fails with `AssertionError: libcuda.so cannot found!`. The symptom is
that everything works until the first timing run.

`<VER>` is the host's driver version, read from `ls $D/libcuda.so.*`. A libcuda
that does not match the loaded kernel driver fails at `cuInit`, so read it
rather than copying a version out of this document. Where the toolkit is
installed, `--gpus all` is simpler and equivalent.

This matters for a reviewer on a shared machine: installing the toolkit needs
root and a Docker daemon restart, and a restart stops every running container
that has no restart policy. The passthrough above avoids both.

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

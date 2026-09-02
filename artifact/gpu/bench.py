"""GraphMend GPU cold-start and steady-state benchmark, paper methodology.

This reproduces the authors' own measurement, not a re-invention of it. The
definitions come from `models/profiling_utils.py` and `cold_start_no_compile.py`
in the reference scripts, and the two matter because a naive
wall-clock timer measures a different quantity and lands near 1x by
construction.

COLD START. Profile from the very FIRST run, with no warmup, and take the
intervals between consecutive `Torch-Compiled Region: 0/0` markers. Interval 1
is the cold run. From it, subtract the union of `backend_compile` spans that
fall inside it. That subtraction is the whole point: merging B+1 subgraphs into
one graph does not reduce total compile work, so compilation is not what
GraphMend removes. What it removes is per-subgraph startup, meaning CUDA-graph
recording, inter-region synchronisation, and eager execution at each break.
Compilation is therefore excluded from BOTH arms.

  cold = (region-0/0 interval 1) - (backend_compile inside that interval)

Measuring total first-call wall clock instead gives roughly 1x on every model,
which is a correct measurement of the wrong thing.

STEADY STATE. The median of the remaining region-0/0 intervals, which are
replays.

CONFIGURATION MATTERS MORE THAN THE METRIC. Three settings in the paper's model
scripts each drag every ratio toward 1.0 when missed: the batch size (about 70%
of VRAM, 837 for MoLFormer, not 8), the input shape (real SMILES padded to about
37 tokens, not 128), and TF32 (allow_tf32 plus matmul_precision("high"), which
this file now sets, worth roughly 2x on Ampere). With all three matched, this
benchmark reproduces the authors' own MoLFormer warm timings to within about 1%:
119.3 ms against their 117.8 ms original, 115.5 ms against their 116.2 ms fixed,
at 1.0 GB against their 0.97 GB peak.

On that matched configuration steady state comes out at 1.033x, and the authors'
own traces give 1.014x, against Table 2's 1.13x. That disagreement is the real
open item, and it is not a mis-configuration artifact: the configuration
reproduces their per-iteration times to 1%.

Both arms load FULL PRETRAINED checkpoints, because latency depends on real
layer counts and widths, and both run under `jac run` with their own jac.toml:
the entry program has to be Jac-compiled or every [Defer] rewrite stays inert
(see paper_eval/README.md). Each arm gets a private inductor and Triton
cache, or the second arm skips codegen and its "cold" run is not cold.

Compilation is `torch.compile(m, backend="inductor", mode="reduce-overhead",
fullgraph=False)`, matching the reference model scripts.

    PYTHONPATH=$PWD python ../artifact/gpu/bench.py t5-small
    PYTHONPATH=$PWD python ../artifact/gpu/bench.py --count t5-small

Known limitation: models that mutate module state inside `forward` cannot be
run under CUDA graphs at all. MoLFormer-XL registers a buffer in `forward`, and
Phi-4-mini does so via `[Where]`'s rewrite of `longrope_frequency_update`. Set
GM_BENCH10_MODE=default for those, and note that the numbers are then not the
paper's CUDA-graph setup.
"""
import argparse, json, os, shutil, statistics, subprocess, sys, tempfile
from datetime import datetime

# The harness owns the path layout; bench.py borrows it so the two cannot drift.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from paper_eval._paths import (  # noqa: E402
    ARTIFACT_ROOT as _ARTIFACT_ROOT,
    ARM_PYTHONPATH as _ARM_PYTHONPATH,
    JACLANG_DIR as _JACLANG_DIR,
)


def _stamp_now():
    """Timestamp for saved trace filenames, matching the reference naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

ARM = "GM_BENCH10_ARM"

# Per-model batch sizes the paper uses on an RTX 3090. It sizes each model to
# about 70%
# of GPU memory and runs the original and fixed variants at the same batch, so a
# comparison at any other batch is measuring a different point. --paper-batch
# selects these; without it the small defaults in build() apply, which are fine
# for break counting and understate the cold-start ratio.
PAPER_BATCH_3090 = {
    "t5-small": 1345,
    "MoLFormer-XL-both10pct": 837,
}
_TOML = ('[dev]\njaclang_source = "{src}"\n\n'
         "[run]\ngraphmend = {on}\ngraphmend_claim_imports = {on}\n")


def _load_weights(m, repo, rev=None, extra_ignorable=()):
    """Real pretrained weights WITHOUT PreTrainedModel.from_pretrained.

    from_pretrained cannot be used here: under `graphmend_claim_imports = true`
    GraphMend claims transformers/modeling_utils.py and the recompiled
    `no_init_weights()` raises UnboundLocalError on its `global _init_weights`.
    The direct constructor path that paper_eval/registry.py already uses is
    unaffected, so the weights are loaded into it by hand.
    """
    import glob, os, torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    d = snapshot_download(repo, revision=rev,
                          allow_patterns=["*.json", "*.safetensors*"])
    files = sorted(glob.glob(os.path.join(d, "*.safetensors")))
    sd = {}
    if files:
        for f in files:
            sd.update(load_file(f))
    else:
        d = snapshot_download(repo, revision=rev,
                              allow_patterns=["*.json", "*.bin"])
        for f in sorted(glob.glob(os.path.join(d, "*.bin"))):
            sd.update(torch.load(f, map_location="cpu", weights_only=True))
    # The checkpoint may be for a task head (MolformerForMaskedLM stores the
    # base model under a "molformer." prefix) while the timed model is the bare
    # base. Pick whichever prefix actually matches the target's keys.
    want = set(m.state_dict())
    best, best_hits = None, len(want & set(sd))
    for pre in {k.split(".")[0] + "." for k in sd if "." in k}:
        cand = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
        hits = len(want & set(cand))
        if hits > best_hits:
            best, best_hits = cand, hits
    # And the mirror case: the checkpoint may be the BASE model while the timed
    # model is a task head, so the TARGET's keys carry a prefix the checkpoint
    # lacks. facebook/bart-base ships BartModel weights (`encoder.layers...`)
    # while BartForConditionalGeneration wants them under `model.`, which shows
    # up as every key missing and every key unexpected at once.
    for pre in {k.split(".")[0] + "." for k in want if "." in k}:
        cand = {pre + k: v for k, v in sd.items()}
        hits = len(want & set(cand))
        if hits > best_hits:
            best, best_hits = cand, hits
    if best is not None:
        sd = best
    res = m.load_state_dict(sd, strict=False)
    # `final_logits_bias` is a BART buffer, not a learned parameter: the
    # constructor registers it as zeros and the published checkpoints omit it,
    # so it is reported missing on every bart-family model while the weights
    # are in fact complete. Zeros is also its correct value, so accepting it
    # changes no output.
    # extra_ignorable: heads a checkpoint legitimately omits. longformer-scico
    # is published without lm_head weights, and the reference loads it with
    # from_pretrained, which initialises them and warns. A randomly initialised
    # head changes no break count and no graph structure, so accepting it here
    # matches the reference rather than loosening the check for everything.
    ignorable = tuple(extra_ignorable) + ("rotary", "inv_freq", "position_ids",
                                          "masked_bias",
                 "final_logits_bias")
    # A tied key (t5's encoder/decoder embed_tokens and lm_head all alias
    # shared.weight) is reported missing but is in fact already loaded, so
    # accept it exactly when it shares storage with a key that did load.
    # state_dict() hands back fresh detached tensors, so identity is useless
    # here; aliased parameters are the ones sharing a storage pointer.
    live = m.state_dict(keep_vars=True)
    miss = set(res.missing_keys)
    loaded_ptrs = {v.data_ptr() for k, v in live.items() if k not in miss}
    missing = [k for k in res.missing_keys
               if not any(t in k for t in ignorable)
               and (k not in live or live[k].data_ptr() not in loaded_ptrs)]
    print(f"# weights: target={len(want)} loaded={len(want) - len(res.missing_keys)} "
          f"missing={len(missing)} unexpected={len(res.unexpected_keys)}",
          file=sys.stderr)
    # A silent all-random load would make every latency number meaningless, so
    # fail loudly rather than quietly benchmarking noise.
    if missing:
        raise SystemExit(f"weight load FAILED: {len(missing)} missing, "
                         f"e.g. {missing[:5]}")
    return m


def _bs(default_b, default_s):
    """Batch/sequence, overridable so the batch can be pushed toward the
    paper's sizing (~70% of GPU memory) instead of a token-sized shape."""
    return (int(os.environ.get("GM_BENCH10_BATCH", default_b)),
            int(os.environ.get("GM_BENCH10_SEQ", default_s)))


# The batch is sized against a plain forward pass, but the timed run compiles
# with mode="reduce-overhead" and captures CUDA graphs, which needs more memory
# than the probe sees. On the largest model that gap is enough to exhaust the
# card at the usual target, so it gets a lower one.
_TARGET = {"Phi-4-mini-instruct": 0.50}


def _auto_batch(model, inputs, torch, key, target=0.70):
    """Size the batch to a target fraction of GPU memory.

    Probes one forward pass at the built-in batch, converts that into a
    per-sample memory cost, inflates it by the same size-tiered generation
    multiplier the reference uses (a single forward with one decoder token
    badly underestimates KV-cache growth), and fills `target` of VRAM.

    Returns None when it cannot measure, in which case the caller keeps the
    built-in batch rather than guessing.
    """
    target = _TARGET.get(key, target)
    import gc
    try:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.empty_cache()
        # Free memory, not just total. Sizing against the card's capacity
        # assumes this process owns it, and on a GPU with anything else
        # resident -- another job, a notebook, a display server -- the estimate
        # exceeds what can be allocated. The verify loop below then halves its
        # way down, rebuilding the model on every failed attempt, which turns a
        # seconds-long probe into minutes and lands on a batch the run never
        # meant to measure.
        free_mem = torch.cuda.mem_get_info()[0]
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated(0)
        probe_b = int(next(iter(inputs.values())).shape[0])
        with torch.inference_mode():
            model(**inputs)
            torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(0)
        torch.cuda.empty_cache()
        gc.collect()

        per_sample = max((peak - baseline) / max(probe_b, 1), 1024)
        model_gb = baseline / (1024 ** 3)
        # The inflation exists for generation, where a KV cache grows over the
        # decode steps and a single-token probe cannot see it. The latency path
        # runs one forward pass and grows nothing, so applying it there divides
        # the budget by up to eighty and fills a few percent of the card
        # instead of the 70% the paper sizes to. The verify loop below halves
        # on a real allocation, so an estimate that is too large is caught.
        if os.environ.get("GM_BENCH10_THROUGHPUT"):
            mult = (80.0 if model_gb > 3.0 else
                    40.0 if model_gb > 1.0 else
                    30.0 if model_gb > 0.3 else 10.0)
        else:
            mult = 1.0
        available = min((total * target) - baseline, free_mem * target)
        if available <= 0:
            return None
        bs = int(available / (per_sample * mult))
        # The cap is a guard against a runaway estimate, not a sizing rule:
        # at 2048 it binds well before 70% on a large card and silently caps
        # the fill. The verify loop below halves on a real allocation, so the
        # ceiling only has to stop something absurd.
        bs = max(1, min(bs, int(os.environ.get("GM_BENCH10_BATCH_CAP", "65536"))))
        print(f"# auto-batch: model {model_gb:.2f} GB, per-sample "
              f"{per_sample / 1024:.1f} KB, multiplier {mult:.0f}x, "
              f"target {target:.0%} of {total / 1024 ** 3:.1f} GB total / "
              f"{free_mem / 1024 ** 3:.1f} GB free -> batch {bs}")
        # Verify rather than trust. The estimate comes from a probe at a small
        # batch, and activation growth is not linear in it, so the number can be
        # too large and the run then dies at allocation time. Halve until a real
        # forward at that batch survives, which is what the reference does
        # before it reports "Verified: batch_size=N fits".
        while bs > 1:
            try:
                os.environ["GM_BENCH10_BATCH"] = str(bs)
                probe_model, probe_inputs = build(key, "cuda")
                with torch.inference_mode():
                    probe_model(**probe_inputs)
                    torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
                del probe_model, probe_inputs
                torch.cuda.empty_cache()
                gc.collect()
                print(f"# auto-batch: verified {bs} fits (peak {peak:.2f} GB)")
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                bs = max(1, bs // 2)
                print(f"# auto-batch: did not fit, retrying at {bs}")
        os.environ.pop("GM_BENCH10_BATCH", None)
        return bs
    except Exception as exc:                      # noqa: BLE001
        print(f"# auto-batch failed ({type(exc).__name__}), keeping default")
        return None


# The BART-family rows of Table 2, whose break count is DTYPE-GATED. The guard
# at transformers/models/bart/modeling_bart.py:568 reads
#
#     if hidden_states.dtype == torch.float16 and (
#         torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()):
#
# and the first conjunct is a static Python bool. In fp32 Dynamo folds it to
# False and never reaches the data-dependent test, so the four DC breaks do not
# exist at all: the same model measures 3 breaks in fp32 and 7 in fp16. The
# reference scripts select fp16 whenever CUDA is present, which is what Table 2
# counts, so these rows are built the same way here: real pretrained weights,
# half precision on the device.
#
# paper_eval/registry.py keeps its fp32 small-config versions of these rows,
# because the paper's correctness claim is that FP32 outputs are bit-identical
# and that is the quantity `output_ok` checks. The two are measuring different
# things on purpose.
_BART_FAMILY = {
    "bart-base":      ("facebook/bart-base",        "BartForConditionalGeneration", 7),
    "bart-large-cnn": ("facebook/bart-large-cnn",   "BartForConditionalGeneration", 7),
    "rebel-large":    ("Babelscape/rebel-large",    "BartForConditionalGeneration", 7),
    "opus-mt-fr-en":  ("Helsinki-NLP/opus-mt-fr-en", "MarianMTModel",               6),
}



# Table 2 text rows, each recipe taken from that row's own reference script:
# (repo, auto class, decoder_input_ids?, prompt).
#
# DTYPE. Every one of these scripts loads with
# `torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32`,
# so they are half on GPU. That is not cosmetic -- BART's break count is
# dtype-gated, and reading it in fp32 loses the data-dependent breaks entirely.
#
# DECODER IDS. The seq2seq rows are fed a single BOS `decoder_input_ids`, the
# token that starts generation, exactly as the BART family is. The two rows
# that take none (Pegasus, longformer) say so in their own fixed_batch().
_TEXT_ROWS = {
    "t5-base":        ("google-t5/t5-base",        "AutoModelForSeq2SeqLM", True),
    "t5-3b":          ("google-t5/t5-3b",          "AutoModelForSeq2SeqLM", True),
    "flan-t5-large":  ("google/flan-t5-large",     "AutoModelForSeq2SeqLM", True),
    "inclusively-reformulation-it5": (
        "E-MIMIC/inclusively-reformulation-it5",   "AutoModelForSeq2SeqLM", True),
    "biogpt":         ("microsoft/biogpt",         "AutoModelForCausalLM",  True),
    "blenderbot-400M-distill": (
        "facebook/blenderbot-400M-distill",        "AutoModelForSeq2SeqLM", True),
    "tiny-random-PegasusForCausalLM": (
        "hf-internal-testing/tiny-random-PegasusForCausalLM",
        "AutoModelForCausalLM",  False),
    # The reference loads this one with AutoModelForMaskedLM; AutoModel drops
    # the head and leaves pooler weights unmatched.
    "longformer-scico": ("allenai/longformer-scico",
                         "AutoModelForMaskedLM", False),
}

# Whisper takes a mel spectrogram, not token ids: the processor turns 30s of
# audio into input_features, and the decoder is started with
# <|startoftranscript|> (50258), one token, as in whisper_*_script.py.
_WHISPER_ROWS = {
    "whisper-base":     "openai/whisper-base",
    "whisper-small":    "openai/whisper-small",
    "whisper-large-v3": "openai/whisper-large-v3",
}


def _text_row(key, device, torch):
    import transformers

    repo, clsname, wants_decoder = _TEXT_ROWS[key]
    cfg = transformers.AutoConfig.from_pretrained(repo, trust_remote_code=True)
    cls = getattr(transformers, clsname)
    m = _load_weights(cls.from_config(cfg, trust_remote_code=True)
                      if hasattr(cls, "from_config") else cls(cfg), repo,
                      extra_ignorable=("lm_head", "pooler")
                      if key == "longformer-scico" else ())
    if device == "cuda":
        m = m.half()
    b, s = _bs(4, 128)
    vocab = getattr(cfg, "vocab_size", None) or 32000
    batch = {
        "input_ids": torch.randint(0, min(vocab, 30000), (b, s), device=device),
        "attention_mask": torch.ones((b, s), dtype=torch.long, device=device),
    }
    if wants_decoder:
        bos = (getattr(cfg, "decoder_start_token_id", None)
               or getattr(cfg, "bos_token_id", None) or 0)
        batch["decoder_input_ids"] = torch.full((b, 1), bos, dtype=torch.long,
                                                device=device)
    return m.to(device).eval(), batch


def _whisper_row(key, device, torch):
    import transformers

    repo = _WHISPER_ROWS[key]
    cfg = transformers.AutoConfig.from_pretrained(repo)
    m = _load_weights(transformers.WhisperForConditionalGeneration(cfg), repo)
    if device == "cuda":
        m = m.half()
    b, _ = _bs(4, 128)
    n_mel = getattr(cfg, "num_mel_bins", 80)
    dt = torch.float16 if device == "cuda" else torch.float32
    # 3000 frames is the processor's fixed 30-second window.
    feats = torch.randn(b, n_mel, 3000, device=device, dtype=dt)
    return m.to(device).eval(), {
        "input_features": feats,
        "decoder_input_ids": torch.full((b, 1), 50258, dtype=torch.long,
                                        device=device)}



def _dummy_image(h=480, w=640):
    """A deterministic RGB image; the processors need a real PIL object."""
    import numpy as np
    from PIL import Image

    rng = np.random.RandomState(0)
    return Image.fromarray(rng.randint(0, 255, (h, w, 3), dtype=np.uint8))


def _to_device(inputs, device, torch):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()}


_GDINO = {
    "grounding-dino-tiny": "IDEA-Research/grounding-dino-tiny",
    "grounding-dino-base": "IDEA-Research/grounding-dino-base",
}


def _build_extra(key, device, torch):
    """Rows needing a processor or Hub remote code. None if not one of them."""
    import transformers

    if key in _GDINO:
        repo = _GDINO[key]
        proc = transformers.AutoProcessor.from_pretrained(repo)
        cfg = transformers.AutoConfig.from_pretrained(repo)
        m = _load_weights(
            transformers.GroundingDinoForObjectDetection(cfg), repo)
        b, _ = _bs(1, 128)
        # fp32 deliberately: the reference records fp16 breaking the fusion
        # layers on this model.
        inputs = proc(images=[_dummy_image()] * b,
                      text=["a cat. a remote control."] * b,
                      return_tensors="pt")
        return m.to(device).eval(), _to_device(inputs, device, torch)

    if key == "layoutlmv3-base":
        repo = "microsoft/layoutlmv3-base"
        proc = transformers.AutoProcessor.from_pretrained(repo, apply_ocr=False)
        cfg = transformers.AutoConfig.from_pretrained(repo)
        m = _load_weights(transformers.LayoutLMv3Model(cfg), repo)
        words = ["hello", "world", "document", "understanding", "test"]
        boxes = [[0, 0, 100, 50], [150, 0, 300, 50], [0, 100, 200, 150],
                 [250, 100, 500, 150], [0, 200, 100, 250]]
        inputs = proc(images=_dummy_image(224, 224), text=words, boxes=boxes,
                      return_tensors="pt", padding=True, truncation=True)
        return m.to(device).eval(), _to_device(inputs, device, torch)

    if key == "chronos-bolt-small":
        import numpy as np
        from chronos.chronos_bolt import ChronosBoltModelForForecasting

        repo = "amazon/chronos-bolt-small"
        cfg = transformers.AutoConfig.from_pretrained(repo)
        m = _load_weights(ChronosBoltModelForForecasting(cfg), repo)
        b, _ = _bs(1, 128)
        rng = np.random.RandomState(42)
        ctx = np.cumsum(rng.randn(b, 128), axis=1).astype("float32")
        return m.to(device).eval(), {
            "context": torch.tensor(ctx, device=device)}

    if key == "Florence-2":
        # registry._florence2, followed exactly. Partial alignment is not enough:
        # with Florence-2-large and pretrained weights the row traces to ONE
        # graph and zero breaks even though every dtype is right (params fp16,
        # hidden_states fp16 into Florence2EncoderLayer), so the fp16 overflow
        # guard never becomes a break and both arms are identical.
        #
        # The registry uses the BASE repo, shrinks the language stack to one
        # encoder and one decoder layer, and builds from config rather than
        # loading pretrained weights. Its docstring records that the count does
        # not move with layer depth, so the shrink costs nothing; what matters
        # is that this is the same model Claim 1 verifies at 7 breaks, so the
        # timed row and the counted row are the same thing.
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        repo = "microsoft/Florence-2-base"
        rev = "5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac"
        cfg = transformers.AutoConfig.from_pretrained(
            repo, trust_remote_code=True, revision=rev)
        cfg.text_config.encoder_layers = 1
        cfg.text_config.decoder_layers = 1
        cls = get_class_from_dynamic_module(
            "modeling_florence2.Florence2ForConditionalGeneration", repo,
            revision=rev)
        m = cls(cfg).half()
        # image_projection is zero-filled by the constructor; the registry seeds
        # it so the vision features reaching the encoder are not identically 0.
        with torch.no_grad():
            torch.nn.init.normal_(m.image_projection, mean=0.0, std=0.02)
        b, s_len = _bs(1, 8)
        return m.to(device).eval(), {
            "input_ids": torch.randint(0, 100, (b, s_len), device=device),
            "decoder_input_ids": torch.randint(0, 100, (b, s_len), device=device),
            "pixel_values": torch.randn(b, 3, 224, 224, device=device).half(),
        }

    if key == "Qwen-Audio-Chat":
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        # Same construction paper_eval/registry.py uses: the class comes from the
        # Hub module by name at a pinned revision, and flash attention is off
        # because the wheel is not installed here. AutoModelForCausalLM does not
        # resolve to the same class for this repo.
        repo = "Qwen/Qwen-Audio-Chat"
        rev = "8b1c0dc720d34da5498f93535f416e3590bf3a71"
        cfg = transformers.AutoConfig.from_pretrained(
            repo, trust_remote_code=True, revision=rev)
        cfg.use_flash_attn = False
        cls = get_class_from_dynamic_module(
            "modeling_qwen.QWenLMHeadModel", repo, revision=rev)
        m = _load_weights(cls(cfg), repo, rev)
        if device == "cuda":
            m = m.half()
        b, _ = _bs(1, 8)
        # Text only, no audio token: the turn shape whose audio-fusion guard
        # costs the two breaks Table 2 reports for this row.
        return m.to(device).eval(), {
            "input_ids": torch.randint(0, 1000, (b, 8), device=device)}

    return None


def build(key, device):
    import torch
    _extra = _build_extra(key, device, torch)
    if _extra is not None:
        return _extra
    if key in _TEXT_ROWS:
        return _text_row(key, device, torch)
    if key in _WHISPER_ROWS:
        return _whisper_row(key, device, torch)
    if key in ("grounding-dino-tiny", "grounding-dino-base"):
        import numpy as np
        from PIL import Image
        from transformers import (AutoConfig, AutoProcessor,
                                  AutoModelForZeroShotObjectDetection)
        repo = ("IDEA-Research/grounding-dino-tiny" if key.endswith("tiny")
                else "IDEA-Research/grounding-dino-base")
        # fp32 deliberately, and the reference script says why: "Grounding DINO
        # has mixed components (Swin backbone + BERT text encoder + fusion).
        # BERT outputs float32, so loading in float16 causes dtype mismatches in
        # fusion layers." So unlike the BART rows this one is NOT dtype-gated.
        #
        # What it IS sensitive to is the input. The break count comes from the
        # real config's size together with a batch built by the model's own
        # processor from a real image: a 480x640 RGB frame becomes a
        # (1, 3, 800, 1066) pixel_values plus a pixel_mask and the tokenised
        # prompt. paper_eval's small-config version of this row synthesises
        # tensors instead and sees 16 breaks; with the processor batch it is 17,
        # which is Table 2's count.
        cfg = AutoConfig.from_pretrained(repo)
        proc = AutoProcessor.from_pretrained(repo)
        m = AutoModelForZeroShotObjectDetection.from_config(cfg)
        b, _ = _bs(1, 0)
        rng = np.random.default_rng(0)
        img = Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8))
        enc = proc(images=[img] * b, text=["a cat. a dog."] * b,
                   return_tensors="pt")
        inputs = {k: (v.to(device) if hasattr(v, "to") else v)
                  for k, v in enc.items()}
        return m.to(device).eval(), inputs
    if key in _BART_FAMILY:
        import transformers
        repo, clsname, _expected = _BART_FAMILY[key]
        cfg = transformers.AutoConfig.from_pretrained(repo)
        m = _load_weights(getattr(transformers, clsname)(cfg), repo)
        # fp16 ALWAYS, not just on CUDA. The reference selects it by device
        # (`torch.float16 if torch.cuda.is_available() else torch.float32`),
        # but what actually determines the break count is the dtype itself,
        # because the guard Dynamo folds is `dtype == torch.float16`. Measured
        # on CPU: fp32 gives 3 breaks and fp16 gives 7, the same as on a GPU.
        # Pinning it here is what lets this row reproduce Table 2 without one.
        m = m.half()
        b, s = _bs(4, 128)
        vocab = getattr(cfg, "vocab_size", 32000)
        # The batch shape matters as much as the dtype here. The reference
        # scripts build `{input_ids, attention_mask, decoder_input_ids}` with
        # the decoder side just ONE token (the BOS that starts generation),
        # not a full-length sequence. Omitting the mask, or feeding a
        # full-length decoder input, traces different code and yields one graph
        # fewer. Reproduced from bart_base_script.py's fixed_batch().
        bos = (getattr(cfg, "decoder_start_token_id", None)
               or getattr(cfg, "bos_token_id", None) or 0)
        return m.to(device).eval(), {
            "input_ids": torch.randint(0, vocab, (b, s), device=device),
            "attention_mask": torch.ones((b, s), dtype=torch.long, device=device),
            "decoder_input_ids": torch.full((b, 1), bos, dtype=torch.long,
                                            device=device)}
    if key == "t5-small":
        from transformers import T5Config, T5ForConditionalGeneration
        cfg = T5Config.from_pretrained("t5-small")
        m = _load_weights(T5ForConditionalGeneration(cfg), "t5-small")
        b, s = _bs(8, 128)
        return m.to(device).eval(), {
            "input_ids": torch.randint(0, 32000, (b, s), device=device),
            "decoder_input_ids": torch.randint(0, 32000, (b, s), device=device)}
    if key == "Phi-4-mini-instruct":
        from transformers import AutoConfig, Phi3ForCausalLM
        repo = "microsoft/Phi-4-mini-instruct"
        cfg = AutoConfig.from_pretrained(repo)
        cfg._attn_implementation = "eager"
        # Pure forward-pass latency: no KV cache to carry state between the
        # iterations (and no cache tensors living in the CUDA graph pool).
        cfg.use_cache = False
        old = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            m = _load_weights(Phi3ForCausalLM(cfg), repo)
        finally:
            torch.set_default_dtype(old)
        b, s = _bs(4, 256)
        return m.to(device).eval(), {
            "input_ids": torch.randint(0, 32000, (b, s), device=device)}
    if key == "MoLFormer-XL-both10pct":
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        repo, rev = "ibm/MoLFormer-XL-both-10pct", "7b12d946c181"
        cfg = AutoConfig.from_pretrained(repo, trust_remote_code=True, revision=rev)
        # deterministic_eval defaults to False, and with it False the linear
        # attention redraws its random Fourier features on EVERY forward:
        #
        #   if not self.deterministic or self.training:
        #       self.orthogonal_random_weights(query.device)   # randn + register_buffer
        #
        # That mutates module state inside the traced region on every call, so
        # the model recompiles after iteration 1 (region 0/0 gives way to 0/1,
        # leaving the cold metric with no second window) and CUDA graphs refuse
        # it outright. It also makes the forward non-deterministic, which is not
        # something to benchmark. True is the config's own documented setting
        # for constant features and is what makes this row measurable.
        cfg.deterministic_eval = True
        cls = get_class_from_dynamic_module(
            "modeling_molformer.MolformerModel", repo, revision=rev)
        m = _load_weights(cls(cfg), repo, rev)
        b, s = _bs(8, 128)
        return m.to(device).eval(), {
            "input_ids": torch.randint(0, 100, (b, s), device=device),
            "attention_mask": torch.ones(b, s, dtype=torch.long, device=device)}
    raise SystemExit(f"unknown model {key}")


def arm():
    import time, torch
    # The paper's model scripts enable TF32 before doing anything else:
    #     torch.backends.cuda.matmul.allow_tf32 = True
    #     torch.set_float32_matmul_precision("high")
    # On Ampere and later this changes fp32 matmul throughput by a large factor,
    # so leaving it off measures a much slower model than the paper measured and
    # drags every ratio toward 1. Inductor warns about it on every run; that
    # warning is the tell. Matching it here is a configuration match, not a
    # thumb on the scale: both arms get it, exactly as in the paper's scripts.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    if os.environ.get("PAPER_EVAL_DIR"):
        sys.path.insert(0, os.environ["PAPER_EVAL_DIR"])
    key = os.environ["GM_MODEL"]
    # GM_BENCH10_DEVICE lets the ON arm be pinned to whatever the OFF arm
    # actually used. Each arm is a separate process, so without the pin a GPU
    # that is busy for one arm and free for the other puts them on different
    # devices, and the output comparison then reports CHANGED for what is only
    # a cpu-vs-cuda floating point difference.
    dev = os.environ.get("GM_BENCH10_DEVICE") or (
        "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model, inputs = build(key, dev)

    # Paper batch sizing: "Batch sizes target about 70% of GPU memory per
    # model; original and fixed models use identical batch sizes and inputs."
    # This mirrors the reference batch sizing in
    # gpu_utils.py, so the batch is chosen by the paper's RULE rather than
    # copied from the paper's MACHINE. On a 24 GB card that yields a smaller
    # batch than the 80 GB host the reference traces came from, which is the
    # point: the rule is what transfers across GPUs, the number is not.
    if (dev == "cuda" and os.environ.get("GM_BENCH10_AUTO_BATCH")
            and not os.environ.get("GM_BENCH10_BATCH")):
        bs = _auto_batch(model, inputs, torch, key)
        if bs and bs > 1:
            os.environ["GM_BENCH10_BATCH"] = str(bs)
            del model, inputs
            torch.cuda.empty_cache()
            torch.manual_seed(0)
            model, inputs = build(key, dev)

    def sync():
        if dev == "cuda":
            torch.cuda.synchronize()

    # One eager forward before anything is compiled, matching the "quick eager
    # sanity check (no compile)" the reference per-model scripts run
    # before profiling. This is not a formality: lazily-populated module caches
    # get filled here, OUTSIDE any captured region. Phi-4's rotary embedding
    # fills `self.long_inv_freq` on first use, and without this warm-up that
    # fill happens inside the CUDA-graph region on the first compiled call, so
    # the module ends up holding a reference into the graph's memory pool and
    # the next replay overwrites it. The failure surfaces far from its cause,
    # as "accessing tensor output of CUDAGraphs that has been overwritten".
    with torch.no_grad():
        model(**inputs)
    if dev == "cuda":
        torch.cuda.synchronize()

    res = {"key": key, "device": dev,
           "dtype": str(next(model.parameters()).dtype),
           "in_shape": list(next(iter(inputs.values())).shape)}

    if os.environ.get("GM_BENCH10_COUNT"):
        graphs = []
        torch._dynamo.reset()
        # `dynamic` is left at its DEFAULT here, deliberately, even though the
        # latency path below pins it to False. The reference counts breaks with
        # `dynamo.explain(model)(**batch)` (gpu_utils.safe_explain), which does
        # not pin it, and on a dynamic-shape model the two disagree: forcing
        # dynamic=False specialises on shape and adds breaks that the reference
        # never sees. grounding-dino-tiny reads 19 with dynamic=False and 17,
        # which is Table 2's count, with the default.
        c = torch.compile(model,
                          backend=lambda gm, ex: (graphs.append(gm), gm.forward)[1])
        with torch.no_grad():
            out = c(**inputs)
        res["graphs"] = len(graphs)
        res["breaks"] = max(0, len(graphs) - 1)
        # Output fingerprint, so the reference rows can carry the same
        # correctness evidence as the small-config rows rather than only a
        # break count. Both arms build from the same seed and the same
        # checkpoint, so a difference here is the transform changing behaviour,
        # which is exactly the thing the paper claims does not happen.
        import hashlib
        t = out if isinstance(out, torch.Tensor) else None
        for attr in ("logits", "last_hidden_state"):
            if hasattr(out, attr):
                t = getattr(out, attr)
                break
        if t is None and isinstance(out, (tuple, list)) and out:
            t = out[0]
        if t is None and hasattr(out, "values"):
            for v in out.values():
                if isinstance(v, torch.Tensor):
                    t = v
                    break
        res["out_hash"] = (
            hashlib.sha256(t.detach().float().cpu().numpy().tobytes()).hexdigest()[:16]
            if isinstance(t, torch.Tensor) else None)
    else:
        cmode = os.environ.get("GM_BENCH10_MODE", "reduce-overhead")
        runs = int(os.environ.get("GM_BENCH10_RUNS", "8"))
        res["compile_mode"] = cmode
        res["runs"] = runs
        import json as _json
        from torch.profiler import profile, ProfilerActivity
        torch._dynamo.reset()
        from torch._dynamo.utils import counters
        counters.clear()
        # Two plain assignments, NOT a conditional expression. GraphMend
        # injects the [Defer] forward pre-hook at a `x = torch.compile(...)`
        # assignment site, and a ternary is not one, so writing this as
        # `c = (torch.compile(..) if .. else torch.compile(..))` leaves the
        # hook uninjected: every deferred call runs inline, the logger breaks
        # survive, and the transformed arm silently measures the untransformed
        # program. The signature of that failure is identical CUDA-graph launch
        # counts between the two arms, which is what this benchmark reports.
        # `dynamic` is left at its default, matching the compile_model() in the
        # reference per-model scripts exactly. Passing dynamic=False
        # specialises on shape and makes a model that mutates module state in
        # forward recompile after iteration 1, which shows up as region "0/1"
        # taking over from "0/0" and leaves the cold window undefined.
        if cmode == "default":
            c = torch.compile(model, backend="inductor", fullgraph=False)
        else:
            c = torch.compile(model, backend="inductor", mode=cmode,
                              fullgraph=False)
        # C10, throughput (paper 5.4). The paper's setup is "Text models
        # generate 100 output tokens with greedy decoding", and throughput is
        # the percentage increase in inference throughput, tokens per second
        # for generative models and samples per second for encoders. That is
        # END TO END, so it includes the decode loop, tokenisation and CPU-side
        # scheduling that GraphMend does not touch. By Amdahl the gain is
        # bounded by the forward pass's share of inference, which is why the
        # paper's own numbers are far smaller than its steady-state ones and
        # why most models here land near zero. See the C10 note in RESULTS.md.
        if os.environ.get("GM_BENCH10_THROUGHPUT"):
            import time as _t
            gen = hasattr(model, "generate") and key != "MoLFormer-XL-both10pct"
            n_tok = int(os.environ.get("GM_BENCH10_GEN_TOKENS", "100"))
            reps = int(os.environ.get("GM_BENCH10_TPUT_REPS", "3"))

            def _once():
                with torch.inference_mode():
                    if gen:
                        return c.generate(**inputs, max_new_tokens=n_tok,
                                          do_sample=False, num_beams=1)
                    return c(**inputs)

            _once()          # compile and capture before timing anything
            sync()
            best = None
            for _ in range(reps):
                t0 = _t.perf_counter()
                _once()
                sync()
                dt = _t.perf_counter() - t0
                best = dt if best is None else min(best, dt)
            batch = int(next(iter(inputs.values())).shape[0])
            units = (batch * n_tok) if gen else batch
            print("GMBENCH10 " + json.dumps({
                "key": key, "throughput": units / best,
                "unit": "tokens/s" if gen else "samples/s",
                "seconds": best, "batch": batch,
                "in_shape": list(next(iter(inputs.values())).shape),
                "dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
            }))
            return

        tf = tempfile.mkstemp(prefix="gmtrace_", suffix=".json")[1]
        # No warmup, and the trace starts before the first call, so interval 1
        # is the genuine cold run: compile plus CUDA-graph capture.
        # acc_events=True is required, not cosmetic. The profiler clears its
        # event buffer at the end of each cycle, and a model whose compilation
        # crosses a cycle boundary loses the region markers entirely: the run
        # completes, the trace exports, and it contains ZERO
        # "Torch-Compiled Region: 0/0" events. Phi-4-mini does exactly that.
        # Older torch does not accept the argument, hence the fallback.
        try:
            _prof = profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                            with_stack=False, acc_events=True)
        except TypeError:
            _prof = profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                            with_stack=False)
        with _prof as prof:
            with torch.inference_mode():
                for _ in range(runs):
                    if cmode != "default":
                        torch.compiler.cudagraph_mark_step_begin()
                    c(**inputs)
                    sync()
        prof.export_chrome_trace(tf)
        with open(tf) as fh:
            evs = _json.load(fh).get("traceEvents", [])
        # Keep the trace when asked, under the same naming the reference
        # scripts use, so `from_trace.py` consumes a reviewer's own trace and
        # the shipped reference traces identically. GraphMend off is the
        # "original" arm and GraphMend on is the "fixed" arm.
        _tdir = os.environ.get("GM_BENCH10_TRACE_DIR")
        if _tdir:
            os.makedirs(_tdir, exist_ok=True)
            # NOT a bare truth test on the variable: the caller sets this to
            # the STRING "0" for the original arm, and "0" is truthy, so both
            # arms were named "fixed" and the original arm's trace was
            # overwritten by the fixed arm that ran after it.
            _armname = ("fixed" if os.environ.get("GM_BENCH10_ON", "") not in
                        ("", "0", "false", "False") else "original")
            _stamp = os.environ.get("GM_BENCH10_STAMP", "run")
            _dest = os.path.join(
                _tdir, f"{os.environ.get('GM_MODEL', 'model')}"
                       f"_trace_{_armname}_{_stamp}.json")
            shutil.copyfile(tf, _dest)
            print(f"# trace written: {_dest}")
        os.unlink(tf)

        # Iteration boundaries. Region 0/N is the FIRST subgraph of an
        # iteration, so consecutive 0/N markers delimit iterations. The authors'
        # scripts match "0/0" exactly, which is right until a model RECOMPILES:
        # Dynamo then issues a second code variant, later iterations arrive as
        # 0/1, and a single 0/0 is left with no window at all. Matching any 0/N
        # and ordering by time is the same measurement on a model that does not
        # recompile and the correct one on a model that does. The variants seen
        # are reported either way, so a recompile stays visible.
        # Iteration boundaries, without assuming the first region is numbered 0.
        # A region name is "Torch-Compiled Region: N/V", N indexing the compiled
        # frame and V the code variant. The FIRST region of each iteration is
        # the lowest N present, and its markers therefore delimit iterations.
        #
        # Two things force this to be computed rather than hardcoded. Phi-4-mini
        # untransformed emits regions 1/0 through 8/0 and NO region 0 at all, so
        # matching "0/0" finds nothing and the model looks unmeasurable. And a
        # model that recompiles issues a second variant, so later iterations
        # arrive as N/1 while N/0 occurs once; matching any variant of the
        # lowest N keeps those iterations. The authors' cold_start_no_compile.py
        # matches "0/0" literally, which is why it cannot measure this model.
        _mark = {}
        for e in evs:
            if not isinstance(e, dict) or "ts" not in e:
                continue
            nm = str(e.get("name", "")).strip()
            if not nm.startswith("Torch-Compiled Region: "):
                continue
            tail = nm[len("Torch-Compiled Region: "):]
            head = tail.split("/")[0]
            if head.isdigit():
                _mark.setdefault(int(head), []).append(e["ts"])
        res["region_index"] = min(_mark) if _mark else None
        reg = sorted(_mark[min(_mark)]) if _mark else []
        res["region_markers"] = len(reg)
        # Any "Region: 0/N" with N>0 is a RECOMPILATION: Dynamo produced another
        # code variant because a guard failed. The metric assumes 0/0 recurs
        # once per iteration, so a recompile leaves no second 0/0 window and the
        # measurement is undefined rather than merely small. Name it, because
        # "fewer than 2 markers" reads like a profiler problem when it is a
        # property of the model.
        import re as _re
        _pre = "Torch-Compiled Region: %s/" % (res.get("region_index"),)
        variants = sorted({_re.sub(r"^Torch-Compiled Region: ", "", str(e.get("name", "")).strip())
                           for e in evs if isinstance(e, dict)
                           and str(e.get("name", "")).strip().startswith(_pre)})
        res["region_variants"] = variants
        res["recompiled"] = len(variants) > 1
        if len(reg) < 2:
            # ONE string literal, no implicit concatenation. Jac's Python
            # ingestion mishandles adjacent string literals (both the plain
            # "a" "b" form and the plain-then-f-string form) and emits the
            # literal source text of the later part, quotes and indentation
            # included. This file runs under `jac run`, so a message split
            # across two literals prints as its own source code. Observed here
            # twice before it was written as a single literal.
            res["error"] = "fewer than 2 region markers (lowest region index %s)" % (res.get("region_index"),)
        else:
            wins = [(reg[i + 1] - reg[i]) / 1000.0 for i in range(len(reg) - 1)]
            lo, hi = reg[0], reg[1]
            spans = sorted((e["ts"], e["ts"] + e["dur"]) for e in evs
                           if isinstance(e, dict) and e.get("name") == "backend_compile"
                           and "ts" in e and "dur" in e and lo <= e["ts"] < hi)
            comp = 0.0
            if spans:
                cs, ce = spans[0]
                for s0, e0 in spans[1:]:
                    if s0 > ce:
                        comp += ce - cs
                        cs, ce = s0, e0
                    else:
                        ce = max(ce, e0)
                comp += ce - cs
            comp /= 1000.0
            res["cold_window_ms"] = wins[0]
            res["cold_compile_ms"] = comp
            res["cold_ms"] = wins[0] - comp
            res["warm_ms"] = (statistics.median(wins[1:]) if len(wins) > 1
                              else float("nan"))
            res["all_windows_ms"] = wins
            # Per-forward counts from the LAST region window, which is a replay.
            win = [e for e in evs if isinstance(e, dict) and e.get("ts", 0) >= reg[-1]]
            res["cudagraph_launches"] = sum(
                1 for e in win if "cudaGraphLaunch" in str(e.get("name", "")))
            res["kernels"] = sum(1 for e in win if e.get("cat", "") == "kernel")
            res["syncs"] = sum(1 for e in win if "Synchronize" in str(e.get("name", "")))
        res["cudagraph_skips"] = int(counters["inductor"].get("cudagraph_skips", 0))
        res["peak_mem_gb"] = (torch.cuda.max_memory_allocated() / 2**30
                              if dev == "cuda" else None)
    print("GMBENCH10 " + json.dumps(res))


def run(key, on, count):
    repo, here = os.getcwd(), os.path.abspath(__file__)
    wd = tempfile.mkdtemp(prefix=f"gmb10_{key}_")
    # Cold start is only cold against an EMPTY compiler cache. TorchInductor and
    # Triton both persist generated kernels to disk, so without a private cache
    # per arm the second arm skips codegen entirely and the off/on comparison
    # measures which arm ran first. A fresh mkdtemp per arm (rather than
    # deleting a shared path) also keeps concurrent runs from racing.
    icache = tempfile.mkdtemp(prefix="gmb10_inductor_")
    tcache = tempfile.mkdtemp(prefix="gmb10_triton_")
    try:
        open(os.path.join(wd, "jac.toml"), "w").write(
            _TOML.format(on="true" if on else "false", src=_JACLANG_DIR))
        shutil.copy(here, os.path.join(wd, "bench10.py"))
        env = dict(os.environ, PYTHONPATH=_ARM_PYTHONPATH,
                   PAPER_EVAL_DIR=_ARTIFACT_ROOT,
                   GM_MODEL=key, TORCHINDUCTOR_CACHE_DIR=icache,
                   TRITON_CACHE_DIR=tcache, **{ARM: "1"})
        # The arm learns which side it is from the environment rather than from
        # the jac.toml it is about to compile, so a saved trace can be named
        # original/fixed the way the reference traces are.
        if on:
            env["GM_BENCH10_ON"] = "1"
        else:
            env.pop("GM_BENCH10_ON", None)
        if os.environ.get("GM_BENCH10_PAPER_BATCH") and key in PAPER_BATCH_3090:
            env["GM_BENCH10_BATCH"] = str(PAPER_BATCH_3090[key])
        if count:
            env["GM_BENCH10_COUNT"] = "1"
        p = subprocess.run([sys.executable, "-m", "jaclang", "run", "bench10.py"],
                           capture_output=True, text=True, env=env, cwd=wd)
        auto_b = None
        for ln in p.stdout.splitlines():
            if ln.startswith("# auto-batch:") and "-> batch " in ln:
                try:
                    auto_b = int(ln.rsplit("-> batch ", 1)[1].strip())
                except ValueError:
                    pass
        for ln in reversed(p.stdout.strip().splitlines()):
            if ln.startswith("GMBENCH10 "):
                res = json.loads(ln[len("GMBENCH10 "):])
                if auto_b:
                    res["auto_batch"] = auto_b
                return res
        # Keep the exception line, not just the tail. A traceback's most useful
        # line is its last "SomeError: message", and slicing the last 500
        # characters drops exactly that whenever the frames below it are deep,
        # which is how a plain CUDA OOM at a large --paper-batch arrives here
        # looking like an anonymous stack fragment.
        import re as _re
        err = (p.stderr.strip() or p.stdout.strip())
        _pat = r"^[A-Za-z_][\w.]*(Error|Exception|Interrupt):"
        exc = next((s for s in (ln.strip() for ln in reversed(err.splitlines()))
                    if _re.match(_pat, s)), "")
        # The pattern above only matches CPython's "SomeError: message". An arm
        # that fails under `jac run` is formatted by the Jac runtime instead, so
        # nothing matches and the tail alone is frames with the reason cut off
        # above them. Keep the head too when there is no matched line, since
        # that is where the reason sits.
        if exc:
            detail = exc + "\n" + err[-500:]
        elif len(err) > 900:
            detail = err[:400] + "\n  ...\n" + err[-500:]
        else:
            detail = err
        return {"key": key, "error": detail}
    finally:
        for d in (wd, icache, tcache):
            shutil.rmtree(d, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--paper-batch", action="store_true",
                    help="use the paper's per-model 3090 batch sizes "
                         "(t5-small 1345, MoLFormer-XL 837). The paper sizes "
                         "each model to about 70%% of GPU memory; the small "
                         "defaults understate the cold-start ratio.")
    ap.add_argument("--count", action="store_true",
                    help="report graph breaks per arm instead of latency")
    ap.add_argument("--json", action="store_true",
                    help="emit the paired off/on results as one JSON object "
                         "instead of the human-readable report. The two "
                         "checked runners beside this file consume it.")
    ap.add_argument("--save-traces", metavar="DIR",
                    help="keep each arm's PyTorch profiler trace under DIR, "
                         "named <model>_trace_<original|fixed>_<stamp>.json, "
                         "the same convention as the reference traces in "
                         "artifact/traces/. Feed them to from_trace.py to "
                         "re-derive cold and steady-state the way the paper "
                         "does, from your own run rather than from ours.")
    ap.add_argument("--runs", type=int, default=None,
                    help="forward passes to profile (default 8). The paper "
                         "profiles seven, giving six inter-marker intervals.")
    ap.add_argument("--throughput", action="store_true",
                    help="measure C10 end-to-end throughput per arm (paper "
                         "5.4): 100 greedy output tokens for generative "
                         "models, forward samples/s for encoders. Expect small "
                         "numbers; the paper's 15%% maximum is Florence-2, "
                         "whose forward pass is a large share of its inference.")
    ap.add_argument("--auto-batch", action="store_true",
                    help="size the batch by the paper's RULE rather than its "
                         "numbers: probe one forward, then fill about 70%% of "
                         "VRAM, as gpu_utils.find_max_batch_size does. Both "
                         "arms get the same batch. Prefer this over "
                         "--paper-batch on a GPU that is not a 24 GB 3090.")
    o = ap.parse_args()
    if getattr(o, "paper_batch", False):
        os.environ["GM_BENCH10_PAPER_BATCH"] = "1"
    if o.runs:
        os.environ["GM_BENCH10_RUNS"] = str(o.runs)
    if o.throughput:
        os.environ["GM_BENCH10_THROUGHPUT"] = "1"
    if o.save_traces:
        os.environ["GM_BENCH10_TRACE_DIR"] = os.path.abspath(o.save_traces)
        # One stamp for the whole invocation, so an original/fixed pair shares
        # it and from_trace.py --dir pairs them unambiguously.
        os.environ.setdefault("GM_BENCH10_STAMP", _stamp_now())
    collected = {}
    for key in o.models:
        if o.auto_batch:
            os.environ["GM_BENCH10_AUTO_BATCH"] = "1"
            os.environ.pop("GM_BENCH10_BATCH", None)
        off = run(key, False, o.count)
        # Pin the ON arm to whatever the OFF arm sized to. The paper requires
        # "original and fixed models use identical batch sizes and inputs",
        # and letting each arm auto-detect is exactly how the reference runs
        # ended up comparing 1252 samples against 1332.
        if o.auto_batch and isinstance(off, dict) and off.get("auto_batch"):
            os.environ["GM_BENCH10_BATCH"] = str(off["auto_batch"])
        on = run(key, True, o.count)
        if o.auto_batch:
            os.environ.pop("GM_BENCH10_BATCH", None)
        if o.json:
            collected[key] = {"off": off, "on": on}
            # Report the row as it lands. The aggregate object is only printed
            # after every model finishes, so a multi-row run used to show
            # nothing at all until the last one was done. This line is not
            # valid JSON, so a reader scanning for the final object skips it.
            if off.get("error") or on.get("error"):
                print(f"ROW {key} ERR off={off.get('error')} "
                      f"on={on.get('error')}", flush=True)
            else:
                ob, nb = off.get("breaks"), on.get("breaks")
                oh, nh = off.get("out_hash"), on.get("out_hash")
                same = "identical" if (oh is not None and oh == nh) else (
                    "CHANGED" if (oh is not None and nh is not None) else "n/a")
                fixed = (ob - nb) if (ob is not None and nb is not None) else None
                pct = f"{round(100 * fixed / ob)}%" if ob else "-"
                print(f"ROW {key:24} breaks {ob} -> {nb}  fixed {fixed} "
                      f"({pct})  {same}", flush=True)
            continue
        if off.get("error") or on.get("error"):
            print(f"{key}: ERR\n  off: {off.get('error')}\n  on: {on.get('error')}")
            continue
        if o.count:
            print(f"{key}: breaks off={off['breaks']} on={on['breaks']} "
                  f"({off['dtype']}, in {off['in_shape']})")
        elif o.throughput:
            po, pn = off.get("throughput"), on.get("throughput")
            if not po or not pn:
                print(f"{key}: no throughput measurement "
                      f"(off={off.get('error')} on={on.get('error')})")
                continue
            print(f"{key}  [{off.get('dtype')}, batch/seq {off.get('in_shape')}]")
            print(f"  throughput        off={po:12.2f} {off.get('unit')}"
                  f"   on={pn:12.2f}   change={(pn / po - 1) * 100:+.2f}%")
        else:
            print(f"{key}  [{off['dtype']}, batch/seq {off['in_shape']}, "
                  f"mode={off.get('compile_mode')}, runs={off.get('runs')}, "
                  f"peak off {off.get('peak_mem_gb') or 0:.1f} GB "
                  f"/ on {on.get('peak_mem_gb') or 0:.1f} GB]")
            if "cold_ms" not in off or "cold_ms" not in on:
                print(f"  no measurement: off={off.get('error')} on={on.get('error')} "
                      f"(region markers off={off.get('region_markers')} "
                      f"on={on.get('region_markers')})")
                continue
            # Cold, with compilation excluded from both arms. This is the
            # paper's definition; the raw window and the compile time it
            # subtracts are printed too, so the subtraction is checkable
            # rather than something the reader has to take on trust.
            # Two cold-start numbers, because the paper and this artifact
            # measured different things and both are worth having.
            #
            # RAW WINDOW is Table 2's metric: the interval between the first two
            # "Torch-Compiled Region: 0/0" markers, original over fixed, with
            # nothing subtracted. Verified against the authors' own 3090
            # MoLFormer traces, where it gives 6200.7 / 250.9 = 24.71x, the
            # value printed in Table 2.
            #
            # NO-COMPILE subtracts backend_compile from both arms, following
            # cold_start_no_compile.py in the authors' repository, on the
            # grounds that merging subgraphs does not reduce total compile work.
            # It is the more conservative number and it is much smaller.
            print(f"  cold, RAW WINDOW  off={off['cold_window_ms']:9.1f}ms "
                  f"on={on['cold_window_ms']:9.1f}ms  "
                  f"speedup={off['cold_window_ms']/on['cold_window_ms']:.2f}x   <- Table 2 metric")
            print(f"  cold, no compile  off={off['cold_ms']:9.1f}ms "
                  f"on={on['cold_ms']:9.1f}ms  "
                  f"speedup={off['cold_ms']/on['cold_ms']:.2f}x   (conservative)")
            # TIME TO STEADY STATE. The two metrics above both anchor on the
            # first region window, which silently undercounts whenever
            # compilation does not finish inside it. On a slower card it often
            # does not: an RTX 3090 running MoLFormer at batch 1136 spends
            # 4108 ms in window 1 and a further 5627 ms in window 2 before
            # reaching a 163 ms steady state, so window 1 alone reports 4.68x
            # where the full cold cost is 8.12x. Summing every window above the
            # steady median is definition-independent and is what a user
            # actually waits through.
            def _to_steady(ws):
                if len(ws) < 3:
                    return None
                steady = statistics.median(ws[2:])
                total = 0.0
                for w in ws:
                    if w <= steady * 1.5:
                        break
                    total += w
                return total or None
            so, sn = _to_steady(off['all_windows_ms']), _to_steady(on['all_windows_ms'])
            if so and sn:
                print(f"  cold, to steady   off={so:9.1f}ms on={sn:9.1f}ms  "
                      f"speedup={so/sn:.2f}x   (all pre-steady windows)")
            print(f"    compile inside  off={off['cold_compile_ms']:9.1f}ms "
                  f"on={on['cold_compile_ms']:9.1f}ms")
            print(f"  warm (median)     off={off['warm_ms']:9.3f}ms "
                  f"on={on['warm_ms']:9.3f}ms  speedup={off['warm_ms']/on['warm_ms']:.3f}x")
            # If the two arms ran at different batch sizes, the raw warm ratio
            # compares unequal amounts of work and is not a speedup. The
            # reference runs this artifact was checked against auto-detected a
            # batch size per arm from free GPU memory, and the two arms landed
            # on different values (1252 against 1332 for opus-mt-fr-en), so
            # normalising per sample is the comparable number.
            _bs = lambda s: (int(str(s).split("x")[0])
                             if str(s).split("x")[0].isdigit() else None)
            bo, bn = _bs(off.get("in_shape")), _bs(on.get("in_shape"))
            if bo and bn and bo != bn:
                print(f"  warm per-sample   off={off['warm_ms']/bo*1000:9.3f}us "
                      f"on={on['warm_ms']/bn*1000:9.3f}us  "
                      f"speedup={(off['warm_ms']/bo)/(on['warm_ms']/bn):.3f}x"
                      f"   <- arms ran at different batch sizes ({bo} vs {bn})")
            if off.get("recompiled") or on.get("recompiled"):
                print(f"  RECOMPILED        off={off.get('recompiled')} "
                      f"(variants {off.get('region_variants')}) "
                      f"on={on.get('recompiled')} (variants {on.get('region_variants')})")
            print(f"  per-forward       launches off={off.get('cudagraph_launches')} "
                  f"on={on.get('cudagraph_launches')}, "
                  f"kernels off={off.get('kernels')} on={on.get('kernels')}, "
                  f"syncs off={off.get('syncs')} on={on.get('syncs')}")
            print(f"  windows off ms: {[round(x,1) for x in off['all_windows_ms']]}")
            print(f"  windows on  ms: {[round(x,1) for x in on['all_windows_ms']]}")
    if o.json:
        print(json.dumps(collected))
    return 0


if os.environ.get(ARM):
    arm()
elif __name__ == "__main__":
    sys.exit(main())

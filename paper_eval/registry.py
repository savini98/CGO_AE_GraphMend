"""Registry of paper-evaluated models: how to build a small instance + inputs,
and which transformers submodule GraphMend should scope-transform.

Small configs (no weight download) are used so break-counting is fast; graph
breaks are structural (code paths), not weight-dependent. torch.manual_seed is
fixed by the runner so the two modes (graphmend on/off) get identical weights,
making the output fingerprints directly comparable for correctness.
"""
import torch


def _t5():
    from transformers import T5Config, T5ForConditionalGeneration
    cfg = T5Config(vocab_size=128, d_model=64, d_ff=128, num_layers=2,
                   num_heads=2, d_kv=32)
    m = T5ForConditionalGeneration(cfg)
    ids = torch.randint(0, 128, (1, 8))
    dec = torch.randint(0, 128, (1, 8))
    return m, {"input_ids": ids, "decoder_input_ids": dec}


def _biogpt():
    from transformers import BioGptConfig, BioGptForCausalLM
    cfg = BioGptConfig(vocab_size=128, hidden_size=64, num_hidden_layers=2,
                       num_attention_heads=2, intermediate_size=128,
                       max_position_embeddings=64)
    m = BioGptForCausalLM(cfg)
    return m, {"input_ids": torch.randint(0, 128, (1, 8))}


def _blenderbot():
    from transformers import BlenderbotConfig, BlenderbotForConditionalGeneration
    cfg = BlenderbotConfig(vocab_size=128, d_model=64, encoder_layers=2,
                           decoder_layers=2, encoder_attention_heads=2,
                           decoder_attention_heads=2, encoder_ffn_dim=128,
                           decoder_ffn_dim=128, max_position_embeddings=64)
    m = BlenderbotForConditionalGeneration(cfg)
    ids = torch.randint(0, 128, (1, 8))
    dec = torch.randint(0, 128, (1, 8))
    return m, {"input_ids": ids, "decoder_input_ids": dec}



def _marian():
    from transformers import MarianConfig, MarianMTModel
    cfg = MarianConfig(vocab_size=128, d_model=64, encoder_layers=2,
                       decoder_layers=2, encoder_attention_heads=2,
                       decoder_attention_heads=2, encoder_ffn_dim=128,
                       decoder_ffn_dim=128, max_position_embeddings=64,
                       decoder_start_token_id=2, pad_token_id=1)
    m = MarianMTModel(cfg)
    ids = torch.randint(3, 128, (1, 8))
    dec = torch.randint(3, 128, (1, 8))
    return m, {"input_ids": ids, "decoder_input_ids": dec}


def _pegasus_causal():
    from transformers import PegasusConfig, PegasusForCausalLM
    cfg = PegasusConfig(vocab_size=128, d_model=64, decoder_layers=2,
                        decoder_attention_heads=2, decoder_ffn_dim=128,
                        max_position_embeddings=64)
    m = PegasusForCausalLM(cfg)
    return m, {"input_ids": torch.randint(0, 128, (1, 8))}


def _phi3_longrope():
    """Phi-4-mini's LongRoPE path -- the paper's Figure 3 worked example.

    `rope_scaling.type = "longrope"` is what routes the forward through
    `longrope_frequency_update`, whose `seq_len > original_max_position_embeddings`
    test is the data-dependent branch [Where] rewrites. Without longrope the
    model takes the default rope path and exhibits no DC break at all.

    short/long_factor must each have length (hidden_size // num_attention_heads) // 2.
    """
    from transformers import Phi3Config, Phi3ForCausalLM
    hidden, heads = 64, 2
    n = (hidden // heads) // 2
    cfg = Phi3Config(
        vocab_size=128, hidden_size=hidden, intermediate_size=128, pad_token_id=0,
        num_hidden_layers=2, num_attention_heads=heads, num_key_value_heads=heads,
        max_position_embeddings=64, original_max_position_embeddings=16,
        rope_scaling={"type": "longrope",
                      "short_factor": [1.0] * n,
                      "long_factor": [2.0] * n},
    )
    m = Phi3ForCausalLM(cfg)
    return m, {"input_ids": torch.randint(0, 128, (1, 8))}


def _molformer():
    """MoLFormer-XL: the paper's [Trap] / validation-guard model (Table 2, VG 5).

    Its `MolformerSelfAttention.forward` holds the Figure 5 pattern verbatim --
    `if not torch.equal(attention_mask, ...): raise ValueError(...)`.

    Two things make this entry unlike the others. It is Hub REMOTE CODE, so it
    needs network access; and the revision is pinned, because the 2026-07
    "Fix deprecated code" commit retargeted the model at a newer transformers
    (it imports `transformers.masking_utils`, absent in the paper's pinned
    4.52.4). `7b12d946c181` is the last revision contemporary with the paper.
    """
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    repo, rev = "ibm/MoLFormer-XL-both-10pct", "7b12d946c181"
    cfg = AutoConfig.from_pretrained(repo, trust_remote_code=True, revision=rev)
    for k, v in [("hidden_size", 64), ("num_hidden_layers", 2),
                 ("num_attention_heads", 2), ("intermediate_size", 128),
                 ("max_position_embeddings", 64)]:
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cls = get_class_from_dynamic_module(
        "modeling_molformer.MolformerModel", repo, revision=rev)
    m = cls(cfg)
    return m, {"input_ids": torch.randint(0, 32, (1, 8)),
               "attention_mask": torch.ones(1, 8, dtype=torch.long)}


def _whisper():
    """Covers whisper-large-v3 / whisper-small / whisper-base (LC rows).

    All three are the same modeling code; the paper lists 3 logger-call breaks
    for each. Whisper takes `input_features`, not `input_ids`.
    """
    from transformers import WhisperConfig, WhisperForConditionalGeneration
    cfg = WhisperConfig(vocab_size=128, d_model=64, encoder_layers=2,
                        decoder_layers=2, encoder_attention_heads=2,
                        decoder_attention_heads=2, encoder_ffn_dim=128,
                        decoder_ffn_dim=128, num_mel_bins=80,
                        max_source_positions=64, max_target_positions=64,
                        decoder_start_token_id=2, pad_token_id=0,
                        bos_token_id=1, eos_token_id=2)
    m = WhisperForConditionalGeneration(cfg)
    feats = torch.randn(1, 80, 128)
    dec = torch.randint(0, 128, (1, 8))
    return m, {"input_features": feats, "decoder_input_ids": dec}


def _bart():
    """Covers bart-large-cnn / bart-base / rebel-large (DC 4 + LC 3 rows)."""
    from transformers import BartConfig, BartForConditionalGeneration
    cfg = BartConfig(vocab_size=128, d_model=64, encoder_layers=2,
                     decoder_layers=2, encoder_attention_heads=2,
                     decoder_attention_heads=2, encoder_ffn_dim=128,
                     decoder_ffn_dim=128, max_position_embeddings=64)
    m = BartForConditionalGeneration(cfg)
    ids = torch.randint(3, 128, (1, 8))
    dec = torch.randint(3, 128, (1, 8))
    return m, {"input_ids": ids, "decoder_input_ids": dec}


def _layoutlmv3():
    """layoutlmv3-base (LC 2)."""
    from transformers import LayoutLMv3Config, LayoutLMv3Model
    # The spatial embeddings are concatenated, so
    # 4*coordinate_size + 2*shape_size must equal hidden_size, and the h/w
    # tables are indexed after a clip to 1023, so max_2d stays at its default.
    cfg = LayoutLMv3Config(vocab_size=128, hidden_size=64, num_hidden_layers=2,
                           num_attention_heads=2, intermediate_size=128,
                           max_position_embeddings=64, max_2d_position_embeddings=1024,
                           coordinate_size=8, shape_size=16,
                           input_size=32, patch_size=16, visual_embed=False)
    m = LayoutLMv3Model(cfg)
    ids = torch.randint(0, 128, (1, 8))
    bbox = torch.zeros(1, 8, 4, dtype=torch.long)
    return m, {"input_ids": ids, "bbox": bbox}


def _longformer():
    """longformer-base-4096 (LC 3 + TI 2; paper fixes 40% -- the 2 tensor.item()
    breaks are declared out of scope, so the expected result here is partial)."""
    from transformers import LongformerConfig, LongformerModel
    cfg = LongformerConfig(vocab_size=128, hidden_size=64, num_hidden_layers=2,
                           num_attention_heads=2, intermediate_size=128,
                           max_position_embeddings=512, attention_window=8)
    m = LongformerModel(cfg)
    # The seq length must NOT be a multiple of attention_window: the logger call
    # lives under `if padding_len > 0`, so a 16-token input with window 8 gives
    # padding_len == 0, the branch never runs, and the LC breaks the paper counts
    # never appear. 20 tokens gives padding_len == 4.
    return m, {"input_ids": torch.randint(0, 128, (1, 20))}


def _grounding_dino():
    """grounding-dino-tiny / -base (DS 3 + DO 3 + DC 11; paper fixes 58%).

    Only the 11 data-dependent control-flow breaks are in scope; the dynamic
    shape and data-dependent operator breaks are the paper's declared
    out-of-scope categories, so a full fix is not the expected result here.
    """
    from transformers import (GroundingDinoConfig, GroundingDinoModel,
                              SwinConfig, BertConfig)
    backbone = SwinConfig(image_size=32, patch_size=4, embed_dim=16,
                          depths=[1, 1], num_heads=[1, 1], window_size=2,
                          out_features=["stage1", "stage2"])
    text = BertConfig(vocab_size=128, hidden_size=64, num_hidden_layers=2,
                      num_attention_heads=2, intermediate_size=128,
                      max_position_embeddings=64)
    cfg = GroundingDinoConfig(backbone_config=backbone, text_config=text,
                              d_model=64, encoder_layers=1, decoder_layers=1,
                              encoder_attention_heads=2, decoder_attention_heads=2,
                              encoder_ffn_dim=128, decoder_ffn_dim=128,
                              num_queries=4, num_feature_levels=2,
                              use_timm_backbone=False)
    m = GroundingDinoModel(cfg)
    return m, {"pixel_values": torch.randn(1, 3, 32, 32),
               "input_ids": torch.randint(0, 128, (1, 8))}


def _chronos_bolt():
    """chronos-bolt-small (LC 6).

    Its repo carries no modeling .py: the code lives in the `chronos` pip
    package (chronos.chronos_bolt), layered on T5. So it is scoped to that
    package, not to transformers_modules, and it is a genuinely separate
    measurement from the plain t5 entry rather than being covered by it.
    """
    from transformers import AutoConfig
    from chronos.chronos_bolt import ChronosBoltModelForForecasting
    cfg = AutoConfig.from_pretrained("amazon/chronos-bolt-small")
    cfg.chronos_config["context_length"] = 64
    cfg.chronos_config["prediction_length"] = 8
    m = ChronosBoltModelForForecasting(cfg)
    return m, {"context": torch.randn(1, 64)}


def _florence2():
    """Florence-2 (DC 7): the largest pure data-dependent control flow row.

    Every one of the breaks is the same site, `Florence2EncoderLayer.forward`:

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):

    Two things have to be right or the row measures nothing.

    HALF PRECISION. In FP32 the dtype test is False and Python short-circuits
    before either `.any()`, so the guard is dead code and the row reads 0
    breaks. It is an FP16 overflow guard and only exists on the FP16 inference
    path -- which is the path the model card's own recipe takes
    (`torch_dtype=torch.float16` whenever CUDA is available) and the one the
    paper measured on its GPUs. Same trap as Phi-4's `longrope` and clap's
    `enable_fusion`.

    FORWARD, NOT GENERATE. `Florence2ForConditionalGeneration.generate` calls
    `self.language_model.generate(...)` and never routes through `self.forward`.
    `torch.compile` only wraps `forward`, and `OptimizedModule.__getattr__`
    hands `generate` straight back off the unwrapped module, so a
    `"call": "generate"` row compiles nothing at all and counts 0 graphs.

    Two things are shrunk, and both were checked against the break count rather
    than assumed. The language stack goes to one encoder and one decoder layer:
    the count does not move with it (1, 3, 6 and 12 encoder layers all report
    7), because the breaks come from how Dynamo splits and resumes around the
    guard, not from one break per layer. The image goes 768 -> 224, which keeps
    the feature map square, as `Florence2VisionModelWithProjection.forward`
    asserts, since both are multiples of the tower's total stride 32.

    The DaViT tower keeps its STOCK DEPTHS. It is the one knob that does move
    the count: a flattened `[1, 1, 1, 1]` tower reports 8 breaks rather than 7.
    """
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    repo = "microsoft/Florence-2-base"
    rev = "5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac"
    cfg = AutoConfig.from_pretrained(repo, trust_remote_code=True, revision=rev)
    cfg.text_config.encoder_layers = 1
    cfg.text_config.decoder_layers = 1
    cls = get_class_from_dynamic_module(
        "modeling_florence2.Florence2ForConditionalGeneration", repo,
        revision=rev)
    m = cls(cfg).half()
    # The Hub code leaves `image_projection` UNINITIALISED:
    #
    #     self.image_projection = nn.Parameter(
    #         torch.empty(image_dim_out, dim_projection))   # modeling_florence2.py:2453
    #
    # `torch.empty` is whatever was in that memory, and transformers'
    # `_init_weights` does not reach a bare nn.Parameter, so nothing ever fills
    # it. It is not dead either -- `x = x @ self.image_projection` (:2519) feeds
    # it straight into the activations.
    #
    # In FP16 that decides whether the encoder's overflow guard sees an inf or a
    # NaN, and that guard is the row's data-dependent branch, so the graph-break
    # count moved between 7 and 8 across otherwise identical runs, taking the
    # output fingerprint with it (measured: 6 runs, breaks 7 once and 8 five
    # times, two distinct output hashes, one per count).
    #
    # Seeding it makes the row a measurement rather than a coin flip. std=0.02
    # is the initialiser range transformers uses for this config family, so this
    # fills the parameter the way the model would have if it had been reached.
    with torch.no_grad():
        torch.nn.init.normal_(m.image_projection, mean=0.0, std=0.02)
    return m, {"input_ids": torch.randint(0, 100, (1, 8)),
               "decoder_input_ids": torch.randint(0, 100, (1, 8)),
               "pixel_values": torch.randn(1, 3, 224, 224).half()}


def _stella():
    """stella-en-400M-v5 (DO + DS, 0% in Table 2). Hub remote code.

    The stock config carries `use_memory_efficient_attention: true` and
    `unpad_inputs: true`, and both are hard xformers dependencies in the remote
    code: `NewAttention.__init__` does `assert self.memory_efficient_attention
    is not None, 'please install xformers'`, and under `unpad_inputs`
    `NewModel.forward` builds its attention bias with
    `xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(...)`. xformers ships no
    macOS wheel and its kernels are CUDA-only.

    The unpad path is where stella's Table 2 breaks live (`torch.nonzero` on the
    flattened mask, `attention_mask.sum(-1).tolist()`, boolean-mask indexing --
    the DO/DS pair). So this builder keeps the stock flags whenever CUDA and
    xformers are both present, which is the only configuration in which that
    break set is reachable, and falls back to the model card's documented
    no-xformers recipe otherwise. The fallback measures a different, smaller
    break set and is NOT the Table 2 row; see README, "stella-en-400M-v5".
    """
    from transformers import AutoConfig, AutoModel
    repo = "NovaSearch/stella_en_400M_v5"
    rev = "ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20"
    cfg = AutoConfig.from_pretrained(repo, trust_remote_code=True, revision=rev)
    for k, v in [("num_hidden_layers", 2), ("hidden_size", 64),
                 ("num_attention_heads", 2), ("intermediate_size", 128),
                 ("vocab_size", 512)]:
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    try:
        import xformers  # noqa: F401
        _xf = True
    except Exception:
        _xf = False
    unpad = bool(_xf and torch.cuda.is_available())
    # Stock config values are both true; only clear them when they cannot run.
    cfg.use_memory_efficient_attention = unpad
    cfg.unpad_inputs = unpad
    m = AutoModel.from_config(cfg, trust_remote_code=True)
    dev = "cuda" if unpad else "cpu"
    m = m.to(dev)
    # token_type_ids is not decoration: the config carries type_vocab_size 2 and
    # stella's tokenizer emits it, and under unpadding it takes its own
    # boolean-mask index (`token_type_ids[attention_mask_bool]`, modeling.py:401).
    # Omitting it leaves that site off the traced path and the row measures 3
    # breaks instead of Table 2's 4.
    return m, {"input_ids": torch.randint(0, 512, (1, 8), device=dev),
               "attention_mask": torch.ones(1, 8, dtype=torch.long, device=dev),
               "token_type_ids": torch.zeros(1, 8, dtype=torch.long, device=dev)}


def _moe_minicpm():
    """moe-minicpm-x4-base (DS 15, 0% in Table 2). Hub remote code."""
    from transformers import AutoConfig, AutoModelForCausalLM
    repo = "babybirdprd/moe-minicpm-x4-base"
    cfg = AutoConfig.from_pretrained(repo, trust_remote_code=True)
    for k, v in [("num_hidden_layers", 2), ("hidden_size", 64),
                 ("num_attention_heads", 2), ("intermediate_size", 128),
                 ("vocab_size", 512)]:
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    return m, {"input_ids": torch.randint(0, 512, (1, 8))}


def _clap():
    """clap-htsat-fused (DS 4, 0% in Table 2).

    Expected to stay unfixed: its breaks are dynamic shape operators, which the
    paper declares out of scope. Registered so the row is measured rather than
    asserted -- a 0% claim still has to show the breaks exist and are declined.
    """
    from transformers import ClapConfig, ClapModel
    cfg = ClapConfig()
    cfg.text_config.vocab_size = 128
    cfg.text_config.hidden_size = 64
    cfg.text_config.num_hidden_layers = 1
    cfg.text_config.num_attention_heads = 2
    cfg.text_config.intermediate_size = 128
    cfg.projection_dim = 32
    # The row is "clap-htsat-FUSED" and the dynamic-shape breaks live in the
    # fusion path, which is off by default. Without this the model traces
    # cleanly and the row reports 0 breaks, measuring nothing.
    cfg.audio_config.enable_fusion = True
    m = ClapModel(cfg)
    n_mel = cfg.audio_config.num_mel_bins
    return m, {"input_ids": torch.randint(0, 128, (1, 8)),
               "attention_mask": torch.ones(1, 8, dtype=torch.long),
               "input_features": torch.randn(1, 4, 1001, n_mel),
               "is_longer": torch.ones(1, 1, dtype=torch.bool)}


def _qwen_audio():
    """Qwen-Audio-Chat. Hub remote code, so it needs network + trust.

    The row exists to measure the audio-fusion guard in `QWenModel.forward`:

        if past_key_values is None and torch.any(input_ids == audio_start_id):

    Table 2 evaluates the text models on text prompts, so the row is text-only:
    `input_ids` carries no audio token and no `audio_info` is supplied, which is
    the shape a chat turn without attached audio has. That is the input on which
    the guard costs two breaks, because Dynamo cannot fold `torch.any` and so
    cannot see that the fusion body is not entered.

    The vocabulary still has to hold `audio_start_id` (155163 in this repo's
    config). Shrinking it to 4096 would put the id outside the range `input_ids`
    can take at all, which makes the guard dead by construction rather than
    merely false on this input, and the row would prove nothing about a branch
    the model can reach. Sizing the vocabulary to the id keeps the guard a real
    data-dependent test that this text-only turn simply does not satisfy.

    Depth is what gets reduced (LM blocks and audio-tower blocks both to 2),
    plus intermediate_size, which is what keeps a 155k-row embedding plus its
    untied lm_head affordable. hidden_size is left alone: the attention
    projection is sized from kv_channels * num_attention_heads independently of
    it, so shrinking it desynchronises the qkv split ("too many values to
    unpack"), and audio.output_dim has to equal hidden_size for the splice at
    modeling_qwen.py:880 to be shape-legal.
    """
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    repo = "Qwen/Qwen-Audio-Chat"
    rev = "8b1c0dc720d34da5498f93535f416e3590bf3a71"
    cfg = AutoConfig.from_pretrained(repo, trust_remote_code=True, revision=rev)
    audio_start_id = cfg.audio["audio_start_id"]
    cfg.num_hidden_layers = 2
    cfg.intermediate_size = 512
    cfg.vocab_size = audio_start_id + 2
    cfg.audio = dict(cfg.audio)
    cfg.audio["n_layer"] = 2
    cfg.use_flash_attn = False
    cls = get_class_from_dynamic_module("modeling_qwen.QWenLMHeadModel", repo,
                                        revision=rev)
    m = cls(cfg)
    return m, {"input_ids": torch.randint(0, 1000, (1, 8))}


MODELS = {
    "t5-small":       {"build": _t5,         "scope": ["transformers.models.t5"]},
    "clap-htsat-fused": {"build": _clap, "scope": ["transformers.models.clap"]},
    "stella-en-400M-v5": {"build": _stella, "scope": ["transformers_modules"]},
    "moe-minicpm-x4-base": {"build": _moe_minicpm, "scope": ["transformers_modules"]},
    # Reads 2 -> 0, 100%, matching Table 2's DC (2). Both breaks are the DC pair
    # at the audio-fusion guard (modeling_qwen.py:760). This is the
    # precondition-conjunct form of [Where]: the true branch dereferences
    # `audio_info` unconditionally, so the rule leads the guard with
    # `audio_info is not None` rather than selecting between a tensor and None.
    "Qwen-Audio-Chat": {"build": _qwen_audio, "scope": ["transformers_modules"]},
    # Same modeling code as t5-base / t5-3b / flan-t5-large /
    # inclusively-reformulation-it5 / chronos-bolt-small, which is why one t5
    # entry stands in for those Table 2 rows.
    "whisper":        {"build": _whisper,    "scope": ["transformers.models.whisper"]},
    "bart":           {"build": _bart,       "scope": ["transformers.models.bart"]},
    # Its breaks are not in the model package at all: they are `warnings.warn`
    # calls in the SHARED transformers.modeling_utils, so the scope must name
    # that module too (same lesson as Phi-4 and modeling_rope_utils).
    "layoutlmv3-base": {"build": _layoutlmv3,
                        "scope": ["transformers.models.layoutlmv3",
                                  "transformers.modeling_utils"]},
    "longformer-base-4096": {"build": _longformer,
                             "scope": ["transformers.models.longformer"]},
    "grounding-dino": {"build": _grounding_dino,
                       "scope": ["transformers.models.grounding_dino"]},
    # Its own package holds the model class, but the breaks are logger calls in
    # transformers' T5 that it builds on, so both have to be in scope.
    "chronos-bolt-small": {"build": _chronos_bolt,
                           "scope": ["chronos", "transformers.models.t5"]},
    # No "call": "generate" -- see _florence2: that path is not compiled at all.
    "Florence-2": {"build": _florence2, "scope": ["transformers_modules"]},
    "biogpt":         {"build": _biogpt,     "scope": ["transformers.models.biogpt"]},
    "blenderbot-400M-distill": {"build": _blenderbot, "scope": ["transformers.models.blenderbot"]},
    "opus-mt-fr-en":  {"build": _marian,        "scope": ["transformers.models.marian"]},
    "PegasusForCausalLM": {"build": _pegasus_causal, "scope": ["transformers.models.pegasus"]},
    # Phi-4-mini exercises [Where]. Its break site is `longrope_frequency_update`
    # in the SHARED top-level `transformers.modeling_rope_utils`, not under
    # `transformers.models.phi3` -- scoping only the model package silently
    # misses it and the model appears to be fixed by [Defer] alone.
    "Phi-4-mini-instruct": {"build": _phi3_longrope,
                            "scope": ["transformers.models.phi3",
                                      "transformers.modeling_rope_utils"]},
    # [Trap]. Opt-in: needs network + trust_remote_code, so it is excluded from
    # the default run (see NETWORK_MODELS). Hub remote code lands under the
    # `transformers_modules.*` namespace; scoping it works because jaclang hooks
    # the source loader as well as sys.meta_path (transformers builds the spec
    # directly, so a meta-path finder alone never sees these modules).
    "MoLFormer-XL-both10pct": {"build": _molformer,
                               "scope": ["transformers_modules"]},
}

# Models the default `python -m paper_eval.run_eval` skips: they download code or
# weights, which the rest of the harness deliberately avoids. Run them by name.
NETWORK_MODELS = {"MoLFormer-XL-both10pct", "chronos-bolt-small", "Florence-2",
                  "Qwen-Audio-Chat", "stella-en-400M-v5", "moe-minicpm-x4-base"}

# Table 2 rows that share modeling code with an entry above. Registering them
# explicitly turns "inferred from shared code" into a measured row. They
# exercise the same code paths as their twin, so a matching result is
# confirmatory rather than independent evidence: what it rules out is a row
# being claimed on an inference nobody ever ran.
_SHARED_CODE_ROWS = {
    "t5-base": "t5-small",
    "t5-3b": "t5-small",
    "flan-t5-large": "t5-small",
    "inclusively-reformulation-it5": "t5-small",
    "whisper-small": "whisper",
    "whisper-base": "whisper",
    "bart-base": "bart",
    "rebel-large": "bart",
    "grounding-dino-base": "grounding-dino",
}
for _row, _twin in _SHARED_CODE_ROWS.items():
    MODELS[_row] = dict(MODELS[_twin])

"""PyTorch BioGPT model."""
from __future__ import annotations
from jaclang.lib.jaclib import __jac_log_emit__, __jac_log_register__
import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...cache_utils import Cache, EncoderDecoderCache
from ...generation import GenerationMixin
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions, SequenceClassifierOutputWithPast, TokenClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, is_torch_flex_attn_available, is_torchdynamo_compiling, logging
from .configuration_biogpt import BioGptConfig
if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask
    from ...integrations.flex_attention import make_flex_block_causal_mask
logger = logging.get_logger(__name__)

class BioGptLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self: Any, num_embeddings: int, embedding_dim: int) -> object:
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self: Any, attention_mask: torch.LongTensor, past_key_values_length: int=0, position_ids: Optional[torch.Tensor]=None) -> object:
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        if position_ids is None:
            attention_mask = attention_mask.long()
            positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1
            position_ids = positions[:, past_key_values_length:]
        return super().forward(position_ids + self.offset)

class BioGptScaledWordEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """

    def __init__(self: Any, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: Optional[float]=1.0) -> object:
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = embed_scale

    def forward(self: Any, input_ids: torch.Tensor) -> object:
        return super().forward(input_ids) * self.embed_scale

class BioGptAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self: Any, embed_dim: int, num_heads: int, dropout: float=0.0, is_decoder: bool=False, bias: bool=True, is_causal: bool=False, config: Optional[BioGptConfig]=None, layer_idx: Optional[int]=None) -> object:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads}).')
        self.scaling = self.head_dim ** (-0.5)
        self.is_decoder = is_decoder
        self.is_causal = is_causal
        self.layer_idx = layer_idx
        if layer_idx is None and self.is_decoder:
            logger.warning_once(f'Instantiating a decoder {self.__class__.__name__} without passing `layer_idx` is not recommended and "\n                "will lead to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "\n                "when creating this class."')
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self: Any, hidden_states: torch.Tensor, key_value_states: Optional[torch.Tensor]=None, past_key_value: Optional[Cache]=None, attention_mask: Optional[torch.Tensor]=None, layer_head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False, cache_position: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        query_states = query_states * self.scaling
        if past_key_value is not None:
            if isinstance(past_key_value, EncoderDecoderCache):
                is_updated = past_key_value.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    curr_past_key_value = past_key_value.cross_attention_cache
                else:
                    curr_past_key_value = past_key_value.self_attention_cache
            else:
                curr_past_key_value = past_key_value
        current_states = key_value_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_value is not None and is_updated:
            key_states = curr_past_key_value.key_cache[self.layer_idx]
            value_states = curr_past_key_value.value_cache[self.layer_idx]
        else:
            key_states = self.k_proj(current_states)
            value_states = self.v_proj(current_states)
            key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
            if past_key_value is not None:
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = curr_past_key_value.update(key_states, value_states, self.layer_idx, {'cache_position': cache_position})
                if is_cross_attention:
                    past_key_value.is_updated[self.layer_idx] = True
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = query_states.reshape(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(f'Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"\n                f" {attn_weights.size()}')
        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :, :key_states.shape[-2]]
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(f'Head mask for a single layer should be of size {(self.num_heads,)}, but is"\n                    f" {layer_head_mask.size()}')
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(f'`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"\n                f" {attn_output.size()}')
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return (attn_output, attn_weights_reshaped, past_key_value)

class BioGptSdpaAttention(BioGptAttention):

    def forward(self: Any, hidden_states: torch.Tensor, key_value_states: Optional[torch.Tensor]=None, past_key_value: Optional[Cache]=None, attention_mask: Optional[torch.Tensor]=None, layer_head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False, cache_position: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        if output_attentions:
            __jac_log_emit__(_gm_log_slot_0, ('BioGptModel is using BioGptSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True` . Falling back to the manual attention"\n                \' implementation, but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.',), {})
            return super().forward(hidden_states, key_value_states=key_value_states, past_key_value=past_key_value, attention_mask=attention_mask, output_attentions=output_attentions, cache_position=cache_position)
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        if past_key_value is not None:
            if isinstance(past_key_value, EncoderDecoderCache):
                is_updated = past_key_value.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    curr_past_key_value = past_key_value.cross_attention_cache
                else:
                    curr_past_key_value = past_key_value.self_attention_cache
            else:
                curr_past_key_value = past_key_value
        current_states = key_value_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_value is not None and is_updated:
            key_states = curr_past_key_value.key_cache[self.layer_idx]
            value_states = curr_past_key_value.value_cache[self.layer_idx]
        else:
            key_states = self.k_proj(current_states)
            value_states = self.v_proj(current_states)
            key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
            if past_key_value is not None:
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = curr_past_key_value.update(key_states, value_states, self.layer_idx, {'cache_position': cache_position})
                if is_cross_attention:
                    past_key_value.is_updated[self.layer_idx] = True
        causal_mask = None
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
        if query_states.device.type == 'cuda' and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        is_causal = True if self.is_causal and causal_mask is None and (tgt_len > 1) else False
        attn_output = torch.nn.functional.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=causal_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return (attn_output, None, past_key_value)
_gm_log_slot_0 = __jac_log_register__(logger.warning_once)
BIOGPT_ATTENTION_CLASSES = {'eager': BioGptAttention, 'sdpa': BioGptSdpaAttention}

class BioGptDecoderLayer(nn.Module):

    def __init__(self: Any, config: BioGptConfig, layer_idx: Optional[int]=None) -> object:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = BIOGPT_ATTENTION_CLASSES[config._attn_implementation](embed_dim=self.embed_dim, num_heads=config.num_attention_heads, dropout=config.attention_probs_dropout_prob, is_decoder=True, is_causal=True, layer_idx=layer_idx)
        self.dropout = config.hidden_dropout_prob
        self.activation_fn = ACT2FN[config.hidden_act]
        self.activation_dropout = config.activation_dropout
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self: Any, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, layer_head_mask: Optional[torch.Tensor]=None, past_key_value: Optional[Cache]=None, output_attentions: Optional[bool]=False, use_cache: Optional[bool]=True, cache_position: Optional[torch.Tensor]=None) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence. It is used to update the
                cache in the correct position and to infer the complete sequence length.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, self_attn_weights, past_key_value = self.self_attn(hidden_states=hidden_states, past_key_value=past_key_value, attention_mask=attention_mask, layer_head_mask=layer_head_mask, output_attentions=output_attentions, cache_position=cache_position)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (past_key_value,)
        return outputs

@auto_docstring
class BioGptPreTrainedModel(PreTrainedModel):
    config_class = BioGptConfig
    base_model_prefix = 'biogpt'
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True

    def _init_weights(self: Any, module: Any) -> object:
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _update_causal_mask(self: Any, attention_mask: Union[torch.Tensor, 'BlockMask'], input_tensor: torch.Tensor, cache_position: torch.Tensor, past_key_values: Cache, output_attentions: bool=False) -> object:
        if self.config._attn_implementation == 'flash_attention_2':
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        if self.config._attn_implementation == 'flex_attention':
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            return attention_mask
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_compilable_cache = past_key_values.is_compileable if past_key_values is not None else False
        if self.config._attn_implementation == 'sdpa' and (not using_compilable_cache) and (not output_attentions):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(attention_mask, inputs_embeds=input_tensor, past_key_values_length=past_seen_tokens, is_training=self.training):
                return None
        dtype = input_tensor.dtype
        sequence_length = input_tensor.shape[1]
        if using_compilable_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else past_seen_tokens + sequence_length + 1
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(attention_mask, sequence_length=sequence_length, target_length=target_length, dtype=dtype, cache_position=cache_position, batch_size=input_tensor.shape[0])
        if self.config._attn_implementation == 'sdpa' and attention_mask is not None and (attention_mask.device.type in ['cuda', 'xpu', 'npu']) and (not output_attentions):
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(attention_mask: torch.Tensor, sequence_length: int, target_length: int, dtype: torch.dtype, cache_position: torch.Tensor, batch_size: int, **kwargs: Any) -> object:
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device)
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)
        return causal_mask

@auto_docstring
class BioGptModel(BioGptPreTrainedModel):

    def __init__(self: Any, config: BioGptConfig) -> object:
        super().__init__(config)
        self.config = config
        self.layerdrop = config.layerdrop
        self.dropout = config.hidden_dropout_prob
        self.embed_dim = config.hidden_size
        self.padding_idx = config.pad_token_id
        embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0
        self.embed_tokens = BioGptScaledWordEmbedding(config.vocab_size, self.embed_dim, self.padding_idx, embed_scale=embed_scale)
        self.embed_positions = BioGptLearnedPositionalEmbedding(config.max_position_embeddings, self.embed_dim)
        self.layers = nn.ModuleList([BioGptDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.gradient_checkpointing = False
        self._use_sdpa = config._attn_implementation == 'sdpa'
        self.post_init()

    def get_input_embeddings(self: Any) -> object:
        return self.embed_tokens

    def set_input_embeddings(self: Any, value: Any) -> object:
        self.embed_tokens = value

    @auto_docstring
    def forward(self: Any, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]]=None, use_cache: Optional[bool]=None, position_ids: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, cache_position: Optional[torch.Tensor]=None, **kwargs: Any) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        if input_ids is not None:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if self.gradient_checkpointing and self.training:
            if use_cache:
                __jac_log_emit__(_gm_log_slot_1, ('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...',), {})
                use_cache = False
        return_legacy_cache = False
        if use_cache and (not isinstance(past_key_values, Cache)):
            return_legacy_cache = True
            __jac_log_emit__(_gm_log_slot_2, ('Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.58.0. "\n                "You should pass an instance of `EncoderDecoderCache` instead, e.g. "\n                "`past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)`.',), {})
            past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)
        batch_size, seq_length = inputs_embeds.size()[:-1]
        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(past_key_values_length, past_key_values_length + seq_length, device=inputs_embeds.device)
        if attention_mask is None and (not is_torchdynamo_compiling()):
            mask_seq_length = past_key_values_length + seq_length
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        self_attn_cache = past_key_values.self_attention_cache if isinstance(past_key_values, EncoderDecoderCache) else past_key_values
        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, self_attn_cache, output_attentions)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        position_ids = self.embed_positions(attention_mask, past_key_values_length, position_ids=position_ids)
        hidden_states = inputs_embeds + position_ids
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = None
        next_decoder_cache = None
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(decoder_layer.__call__, hidden_states, causal_mask, head_mask[idx] if head_mask is not None else None, None, output_attentions, use_cache, cache_position)
            else:
                layer_outputs = decoder_layer(hidden_states, attention_mask=causal_mask, layer_head_mask=head_mask[idx] if head_mask is not None else None, past_key_value=past_key_values, output_attentions=output_attentions, use_cache=use_cache, cache_position=cache_position)
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        hidden_states = self.layer_norm(hidden_states)
        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = past_key_values.to_legacy_cache()
        if not return_dict:
            return tuple((v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions] if v is not None))
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=next_cache, hidden_states=all_hidden_states, attentions=all_self_attns, cross_attentions=all_cross_attentions)
_gm_log_slot_2 = __jac_log_register__(logger.warning_once)
_gm_log_slot_1 = __jac_log_register__(logger.warning_once)

@auto_docstring(custom_intro='\n    BioGPT Model with a `language modeling` head on top for CLM fine-tuning.\n    ')
class BioGptForCausalLM(BioGptPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ['output_projection.weight']

    def __init__(self: Any, config: Any) -> object:
        super().__init__(config)
        self.biogpt = BioGptModel(config)
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_output_embeddings(self: Any) -> object:
        return self.output_projection

    def set_output_embeddings(self: Any, new_embeddings: Any) -> object:
        self.output_projection = new_embeddings

    @auto_docstring
    def forward(self: Any, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, position_ids: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, cache_position: Optional[torch.Tensor]=None, **kwargs: Any) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        '''r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """'''
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.biogpt(input_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, past_key_values=past_key_values, use_cache=use_cache, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, cache_position=cache_position)
        sequence_output = outputs[0]
        prediction_scores = self.output_projection(sequence_output)
        lm_loss = None
        if labels is not None:
            lm_loss = self.loss_function(prediction_scores, labels, vocab_size=self.config.vocab_size, **kwargs)
        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return (lm_loss,) + output if lm_loss is not None else output
        return CausalLMOutputWithCrossAttentions(loss=lm_loss, logits=prediction_scores, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions, cross_attentions=outputs.cross_attentions)

    @staticmethod
    def _reorder_cache(past_key_values: Any, beam_idx: Any) -> object:
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple((past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)),)
        return reordered_past

@auto_docstring
class BioGptForTokenClassification(BioGptPreTrainedModel):

    def __init__(self: Any, config: Any) -> object:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.biogpt = BioGptModel(config)
        if hasattr(config, 'classifier_dropout') and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        else:
            classifier_dropout = config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    @auto_docstring
    def forward(self: Any, input_ids: Optional[torch.LongTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, TokenClassifierOutput]:
        '''r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """'''
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.biogpt(input_ids, past_key_values=past_key_values, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = transformer_outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels))
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return (loss,) + output if loss is not None else output
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)

@auto_docstring(custom_intro='\n    The BioGpt Model transformer with a sequence classification head on top (linear layer).\n\n    [`BioGptForSequenceClassification`] uses the last token in order to do the classification, as other causal models\n    (e.g. GPT-2) do.\n\n    Since it does classification on the last token, it is required to know the position of the last token. If a\n    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If\n    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the\n    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in\n    each row of the batch).\n    ')
class BioGptForSequenceClassification(BioGptPreTrainedModel):

    def __init__(self: Any, config: BioGptConfig) -> object:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.biogpt = BioGptModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.post_init()

    @auto_docstring
    def forward(self: Any, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        '''r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """'''
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.biogpt(input_ids, past_key_values=past_key_values, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)
        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]
        if self.config.pad_token_id is None:
            sequence_length = -1
        elif input_ids is not None:
            sequence_length = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
        else:
            sequence_length = -1
            __jac_log_emit__(_gm_log_slot_3, (f'{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "\n                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"',), {})
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_length]
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = 'regression'
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = 'single_label_classification'
                else:
                    self.config.problem_type = 'multi_label_classification'
            if self.config.problem_type == 'regression':
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == 'single_label_classification':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == 'multi_label_classification':
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        return SequenceClassifierOutputWithPast(loss=loss, logits=pooled_logits, past_key_values=transformer_outputs.past_key_values, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)

    def get_input_embeddings(self: Any) -> object:
        return self.biogpt.embed_tokens

    def set_input_embeddings(self: Any, value: Any) -> object:
        self.biogpt.embed_tokens = value
_gm_log_slot_3 = __jac_log_register__(logger.warning_once)
__all__ = ['BioGptForCausalLM', 'BioGptForTokenClassification', 'BioGptForSequenceClassification', 'BioGptModel', 'BioGptPreTrainedModel']
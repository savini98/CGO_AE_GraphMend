""" PyTorch Molformer model."""
from __future__ import annotations
from jaclang.lib.jaclib import __jac_log_emit__, __jac_log_register__, __jac_tensor_eq_assert__
import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, MaskedLMOutput, SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_molformer import MolformerConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'ibm/MoLFormer-XL-both-10pct'
_CONFIG_FOR_DOC = 'MolformerConfig'
MOLFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = ['ibm/MoLFormer-XL-both-10pct']

def rotate_half(x: Any) -> object:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: Any, k: Any, cos: Any, sin: Any, position_ids: Any) -> object:
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return (q_embed, k_embed)

class MolformerRotaryEmbedding(nn.Module):

    def __init__(self: Any, dim: Any, max_position_embeddings: Any=2048, base: Any=10000, device: Any=None) -> object:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self: Any, seq_len: Any, device: Any, dtype: Any) -> object:
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos().to(dtype), persistent=False)
        self.register_buffer('sin_cached', emb.sin().to(dtype), persistent=False)

    def forward(self: Any, x: Any, seq_len: Any=None) -> object:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (self.cos_cached[:seq_len].to(dtype=x.dtype), self.sin_cached[:seq_len].to(dtype=x.dtype))

class MolformerEmbeddings(nn.Module):
    """Construct the embeddings from word embeddings."""

    def __init__(self: Any, config: Any) -> object:
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.dropout = nn.Dropout(config.embedding_dropout_prob)

    def forward(self: Any, input_ids: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        embeddings = inputs_embeds
        embeddings = self.dropout(embeddings)
        return embeddings

class MolformerFeatureMap(nn.Module):

    def __init__(self: Any, config: Any) -> object:
        super().__init__()
        self.query_size = config.hidden_size // config.num_attention_heads
        self.num_components = config.num_random_features
        self.orthogonal_random_weights()
        if isinstance(config.feature_map_kernel, str):
            self.kernel = ACT2FN[config.feature_map_kernel]
        else:
            self.kernel = config.feature_map_kernel
        self.deterministic = config.deterministic_eval

    def orthogonal_random_weights(self: Any, device: Any=None) -> object:
        num_batches = math.ceil(self.num_components / self.query_size)

        def orthogonal_batch(size: Any) -> object:
            block = torch.randn(size, size, device=device)
            norms = torch.linalg.norm(block, dim=1).unsqueeze(0)
            Q, _ = torch.linalg.qr(block)
            return Q * norms
        random_weights = torch.cat([orthogonal_batch(self.query_size) for _ in range(num_batches)], dim=1)
        random_weights = random_weights[:, :self.num_components]
        self.register_buffer('weight', random_weights)

    def forward(self: Any, query: Any, key: Any) -> object:
        if not self.deterministic or self.training:
            self.orthogonal_random_weights(query.device)
        query = torch.matmul(query, self.weight)
        key = torch.matmul(key, self.weight)
        return (self.kernel(query), self.kernel(key))

class MolformerSelfAttention(nn.Module):

    def __init__(self: Any, config: Any) -> object:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and (not hasattr(config, 'embedding_size')):
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention "\n                f"heads ({config.num_attention_heads})')
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.eps = config.linear_attention_eps
        self.rotary_embeddings = MolformerRotaryEmbedding(dim=self.attention_head_size, max_position_embeddings=config.max_position_embeddings)
        self.feature_map = MolformerFeatureMap(config)

    def transpose_for_scores(self: Any, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self: Any, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=False) -> Tuple[torch.Tensor]:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        kv_seq_len = key_layer.shape[-2]
        cos, sin = self.rotary_embeddings(value_layer, seq_len=kv_seq_len)
        query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin, position_ids)
        query_layer, key_layer = self.feature_map(query_layer, key_layer)
        if attention_mask is not None:
            attention_mask = (attention_mask == 0).to(attention_mask.dtype)
            per_query_attn = attention_mask[:, 0, -1]
            per_query_extended = per_query_attn[:, None, None, :]
            __jac_tensor_eq_assert__(attention_mask, per_query_extended, '[[GM-TRAP ValueError]]MolformerSelfAttention does not support arbitrary 3D attention. attention_mask must be 2D (i.e., [batch size, sequence length])[[/GM-TRAP]]')
            key_layer = key_layer * per_query_attn[:, None, -kv_seq_len:, None]
        key_value = torch.matmul(key_layer.transpose(-1, -2), value_layer)
        norm = torch.matmul(query_layer, key_layer.sum(dim=-2).unsqueeze(-1)).clamp(min=self.eps)
        context_layer = torch.matmul(query_layer, key_value) / norm
        if head_mask is not None:
            context_layer = context_layer * head_mask
        if output_attentions:
            __jac_log_emit__(_gm_log_slot_0, ('Outputting attentions in linear attention negates the efficiency gains! Only use for visualization/debugging.',), {})
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            if attention_mask is not None:
                attention_scores = attention_scores * attention_mask
            attention_probs = nn.functional.normalize(attention_scores, p=1, dim=-1, eps=self.eps)
            if head_mask is not None:
                attention_probs = attention_probs * head_mask
            context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs
_gm_log_slot_0 = __jac_log_register__(logger.warning)

class MolformerSelfOutput(nn.Module):

    def __init__(self: Any, config: Any) -> object:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self: Any, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class MolformerAttention(nn.Module):

    def __init__(self: Any, config: Any) -> object:
        super().__init__()
        self.self = MolformerSelfAttention(config)
        self.output = MolformerSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self: Any, heads: Any) -> object:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads)
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self: Any, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=False) -> Tuple[torch.Tensor]:
        self_outputs = self.self(hidden_states, attention_mask, position_ids, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class MolformerIntermediate(nn.Module):

    def __init__(self: Any, config: Any) -> object:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self: Any, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class MolformerOutput(nn.Module):

    def __init__(self: Any, config: Any) -> object:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self: Any, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class MolformerLayer(nn.Module):

    def __init__(self: Any, config: Any) -> object:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = MolformerAttention(config)
        self.intermediate = MolformerIntermediate(config)
        self.output = MolformerOutput(config)

    def forward(self: Any, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=False) -> Tuple[torch.Tensor]:
        self_attention_outputs = self.attention(hidden_states, attention_mask, position_ids, head_mask, output_attentions=output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        layer_output = apply_chunking_to_forward(self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output)
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self: Any, attention_output: Any) -> object:
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class MolformerEncoder(nn.Module):

    def __init__(self: Any, config: Any) -> object:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([MolformerLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self: Any, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=False, output_hidden_states: Optional[bool]=False, return_dict: Optional[bool]=True) -> Union[Tuple[torch.Tensor], BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module: Any) -> object:

                    def custom_forward(*inputs: Any) -> object:
                        return module(*inputs, output_attentions)
                    return custom_forward
                layer_outputs = torch.utils.checkpoint.checkpoint(create_custom_forward(layer_module), hidden_states, attention_mask, position_ids, layer_head_mask)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, position_ids, layer_head_mask, output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions)

class MolformerPredictionHeadTransform(nn.Module):

    def __init__(self: Any, config: Any) -> object:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self: Any, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class MolformerLMPredictionHead(nn.Module):

    def __init__(self: Any, config: Any) -> object:
        super().__init__()
        self.transform = MolformerPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self: Any, hidden_states: Any) -> object:
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class MolformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = MolformerConfig
    base_model_prefix = 'molformer'
    supports_gradient_checkpointing = True
    _no_split_modules = ['MolformerEmbeddings', 'MolformerSelfAttention']

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

    def _set_gradient_checkpointing(self: Any, module: Any, value: Any=False) -> object:
        if isinstance(module, MolformerEncoder):
            module.gradient_checkpointing = value

def masked_avg_pool1d(hidden_states: Any, attention_mask: Any, eps: Any=1e-09) -> object:
    attention_mask = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
    sum_embeddings = torch.sum(hidden_states * attention_mask, dim=1)
    sum_mask = torch.clamp(attention_mask.sum(dim=1), min=eps)
    embedding = sum_embeddings / sum_mask
    return embedding
MOLFORMER_START_DOCSTRING = '\n    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use\n    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and\n    behavior.\n\n    Parameters:\n        config ([`MolformerConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
MOLFORMER_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `({0})`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.n_positions - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n"

@add_start_docstrings('The bare Molformer Model transformer outputting raw hidden-states without any specific head on top.', MOLFORMER_START_DOCSTRING, '\n        add_pooling_layer (`bool`, *optional*, defaults to `True`):\n            Whether or not to apply pooling layer.\n    ')
class MolformerModel(MolformerPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention).
    """

    def __init__(self: Any, config: Any, add_pooling_layer: Any=True) -> object:
        super().__init__(config)
        self.config = config
        self.embeddings = MolformerEmbeddings(config)
        self.encoder = MolformerEncoder(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = masked_avg_pool1d if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self: Any) -> object:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self: Any, value: Any) -> object:
        self.embeddings.word_embeddings = value

    def _prune_heads(self: Any, heads_to_prune: Any) -> object:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(MOLFORMER_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(self: Any, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[BaseModelOutputWithPooling, Tuple[torch.Tensor]]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        embedding_output = self.embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask, position_ids=position_ids, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        sequence_output = self.LayerNorm(sequence_output)
        pooled_output = self.pooler(sequence_output, attention_mask) if self.pooler is not None else None
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

@add_start_docstrings('Molformer Model with a `language modeling` head on top.', MOLFORMER_START_DOCSTRING)
class MolformerForMaskedLM(MolformerPreTrainedModel):
    _tied_weights_keys = ['lm_head.decoder.weight']

    def __init__(self: Any, config: Any) -> object:
        super().__init__(config)
        if config.is_decoder:
            logger.warning('If you want to use `MolformerForMaskedLM` make sure `config.is_decoder=False` for "\n                "bi-directional self-attention.')
        self.molformer = MolformerModel(config, add_pooling_layer=False)
        self.lm_head = MolformerLMPredictionHead(config)
        self.post_init()

    def get_output_embeddings(self: Any) -> object:
        return self.lm_head.decoder

    def set_output_embeddings(self: Any, new_embeddings: Any) -> object:
        self.lm_head.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(MOLFORMER_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC, mask='P<mask>')
    def forward(self: Any, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, encoder_hidden_states: Optional[torch.FloatTensor]=None, encoder_attention_mask: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[MaskedLMOutput, Tuple[torch.Tensor]]:
        '''r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """'''
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.molformer(input_ids, attention_mask=attention_mask, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)
        masked_lm_loss = None
        if labels is not None:
            labels = labels.to(prediction_scores.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (masked_lm_loss,) + output if masked_lm_loss is not None else output
        return MaskedLMOutput(loss=masked_lm_loss, logits=prediction_scores, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

class MolformerClassificationHead(nn.Module):
    """Head for sequence-level classification tasks."""

    def __init__(self: Any, config: Any) -> object:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.classifier_dropout_prob if config.classifier_dropout_prob is not None else config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        if isinstance(config.hidden_act, str):
            self.classifier_act_fn = ACT2FN[config.hidden_act]
        else:
            self.classifier_act_fn = config.hidden_act
        self.skip_connection = config.classifier_skip_connection

    def forward(self: Any, pooled_output: Any) -> object:
        hidden_state = self.dense(pooled_output)
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.classifier_act_fn(hidden_state)
        if self.skip_connection:
            hidden_state = residual = hidden_state + pooled_output
        hidden_state = self.dense2(hidden_state)
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.classifier_act_fn(hidden_state)
        if self.skip_connection:
            hidden_state = hidden_state + residual
        logits = self.out_proj(hidden_state)
        return logits

@add_start_docstrings('\n    Molformer Model transformer with a sequence classification/regression head on top (a linear layer on top of the\n    pooled output) e.g. for MoleculeNet tasks.\n    ', MOLFORMER_START_DOCSTRING)
class MolformerForSequenceClassification(MolformerPreTrainedModel):

    def __init__(self: Any, config: Any) -> object:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.molformer = MolformerModel(config, add_pooling_layer=True)
        self.classifier = MolformerClassificationHead(config)
        self.post_init()

    @add_start_docstrings_to_model_forward(MOLFORMER_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self: Any, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        '''r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """'''
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.molformer(input_ids, attention_mask=attention_mask, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
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
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == 'single_label_classification':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == 'multi_label_classification':
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
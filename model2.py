import math
import warnings
from typing import Optional, Tuple, List, Union, Callable

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, PretrainedConfig

# --- 헬퍼 함수 및 상수 ---

# 간단한 logger warning_once 구현 (없으면 print로 대체)
class SimpleLogger:
    _warned = set()
    @classmethod
    def warning_once(cls, msg):
        if msg not in cls._warned:
            print(f"Warning: {msg}")
            cls._warned.add(msg)
logger = SimpleLogger()

# ACT2FN 맵 (BART에서 쓰는 활성화 함수 매핑)
ACT2FN = {
    "gelu": nn.functional.gelu,
    "relu": nn.functional.relu,
    "silu": nn.functional.silu,
    "gelu_new": nn.functional.gelu,  # gelu_new와 gelu 동일 처리 (간단히)
}

# Cache, EncoderDecoderCache, BaseModelOutputWithPastAndCrossAttentions 등은
# 본격적인 캐싱/출력 객체이므로 간략하게 None 또는 튜플로 처리(실제 학습 시 개선 필요)
Cache = object
EncoderDecoderCache = object

class BaseModelOutputWithPastAndCrossAttentions:
    def __init__(self, last_hidden_state, past_key_values, hidden_states=None, attentions=None, cross_attentions=None):
        self.last_hidden_state = last_hidden_state
        self.past_key_values = past_key_values
        self.hidden_states = hidden_states
        self.attentions = attentions
        self.cross_attentions = cross_attentions

# Dummy is_torchdynamo_compiling 함수
def is_torchdynamo_compiling():
    return False

# --- 임베딩 클래스 ---

class BartScaledWordEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_dim, padding_idx, embed_scale=1.0):
        super().__init__(vocab_size, embed_dim, padding_idx=padding_idx)
        self.embed_scale = embed_scale

    def forward(self, input_ids):
        return super().forward(input_ids) * self.embed_scale

class BartLearnedPositionalEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0, std=0.02)

    def forward(self, input_ids, past_key_values_length=0, position_ids=None):
        seq_len = input_ids.shape[1]
        if position_ids is None:
            position_ids = torch.arange(past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=input_ids.device)
        return self.weight[position_ids]

# --- BartAttention ---

class BartAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[PretrainedConfig] = None,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got embed_dim: {self.embed_dim} and num_heads: {num_heads})."
            )
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal
        self.layer_idx = layer_idx
        if layer_idx is None and self.is_decoder:
            logger.warning_once(
                f"Instantiating a decoder {self.__class__.__name__} without passing layer_idx is not recommended."
            )

        # 각 Linear의 입력과 출력 차원을 embed_dim으로 맞춤
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, tgt_len, embed_dim = hidden_states.size()
        src_len = key_value_states.size(1) if key_value_states is not None else tgt_len

        # query shape: (bsz, num_heads, tgt_len, head_dim)
        query = self.q_proj(hidden_states).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        if key_value_states is not None:
            # key, value shape: (bsz, num_heads, src_len, head_dim)
            key = self.k_proj(key_value_states).view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
            value = self.v_proj(key_value_states).view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            key = self.k_proj(hidden_states).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
            value = self.v_proj(hidden_states).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        # scores shape: (bsz, num_heads, tgt_len, src_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scaling

        if attention_mask is not None:
            # attention_mask shape: (bsz, 1, 1, src_len)
            scores = scores + attention_mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # attn_output shape: (bsz, num_heads, tgt_len, head_dim)
        attn_output = torch.matmul(attn_weights, value)

        # (bsz, tgt_len, num_heads, head_dim) 으로 transpose 후 reshape
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        if output_attentions:
            return attn_output, attn_weights, past_key_value
        else:
            return attn_output, None, past_key_value




# --- BartDecoderLayer ---

class BartDecoderLayer(nn.Module):
    def __init__(self, config: PretrainedConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
            layer_idx=layer_idx,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
            layer_idx=layer_idx,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states, self_attn_weights, past_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states, cross_attn_weights, past_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                cache_position=cache_position,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        if use_cache:
            outputs += (past_key_value,)
        return outputs

# --- BartDecoder ---

class BartDecoder(nn.Module):
    def __init__(self, config: PretrainedConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = BartScaledWordEmbedding(config.vocab_size, config.d_model, self.padding_idx, embed_scale=embed_scale)
        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = BartLearnedPositionalEmbedding(config.max_position_embeddings, config.d_model)
        self.layers = nn.ModuleList([BartDecoderLayer(config, layer_idx=i) for i in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _update_causal_mask(self, attention_mask, inputs_embeds, cache_position, self_attn_cache):
        seq_len = attention_mask.size(1)  # 시퀀스 길이

        # causal mask 생성 (상삼각 행렬)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=attention_mask.device), diagonal=1)
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))  # [seq_len, seq_len]

        # 차원 확장
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [bsz, 1, 1, seq_len]

        # 더하기 (broadcasting)
        combined_mask = attention_mask + causal_mask  # [bsz, 1, seq_len, seq_len]
        return combined_mask

    def _update_cross_attn_mask(self, encoder_hidden_states, encoder_attention_mask, input_shape, inputs_embeds):
        if encoder_attention_mask is None:
            return None
        return encoder_attention_mask.unsqueeze(1).unsqueeze(2)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else True

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once("use_cache=True is incompatible with gradient checkpointing. Setting use_cache=False...")
                use_cache = False

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            input_shape = input_ids.size()
            inputs_embeds = self.embed_tokens(input_ids)
        else:
            input_shape = inputs_embeds.size()[:-1]

        batch_size, seq_length = input_shape
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = len(past_key_values[0][0]) if past_key_values else 0

        if cache_position is None:
            cache_position = torch.arange(past_key_values_length, past_key_values_length + seq_length, device=inputs_embeds.device)

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=inputs_embeds.device)

        # causal mask 업데이트
        attention_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values)

        # cross attention mask 업데이트
        encoder_attention_mask = self._update_cross_attn_mask(encoder_hidden_states, encoder_attention_mask, input_shape, inputs_embeds)

        hidden_states = inputs_embeds + self.embed_positions(input_ids, past_key_values_length, position_ids=cache_position)
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                layer_head_mask=head_mask[idx] if head_mask is not None else None,
                cross_attn_layer_head_mask=cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                past_key_value=past_key_values[idx] if past_key_values is not None else None,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[3 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            outputs = tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions] if v is not None)
            return outputs

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

# --- BartConfig ---

class BartConfig(PretrainedConfig):
    model_type = "bart"

    def __init__(
        self,
        vocab_size=50265,
        max_position_embeddings=1024,
        encoder_layers=12,
        encoder_ffn_dim=4096,
        encoder_attention_heads=16,
        decoder_layers=12,
        decoder_ffn_dim=4096,
        decoder_attention_heads=16,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        activation_function="gelu",
        d_model=1024,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        classifier_dropout=0.0,
        scale_embedding=False,
        use_cache=True,
        num_labels=3,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        is_encoder_decoder=True,
        decoder_start_token_id=2,
        forced_eos_token_id=2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = scale_embedding

        super().__init__(
            num_labels=num_labels,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )

# --- CorrectionModel ---

class CorrectionModel(nn.Module):
    def __init__(self, encoder_name="beomi/kcbert-base", num_decoder_layers=6, vocab_size=32000):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(encoder_name)
        self.encoder = BertModel.from_pretrained(encoder_name)

        self.config = BartConfig()
        self.config.d_model = self.encoder.config.hidden_size
        self.config.decoder_layers = num_decoder_layers
        self.config.vocab_size = self.tokenizer.vocab_size
        self.config.decoder_attention_heads = 12

        self.decoder = BartDecoder(self.config)
        self.output_projection = nn.Linear(self.config.d_model, self.config.vocab_size)

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask=None):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        memory = encoder_outputs.last_hidden_state

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=memory,
            encoder_attention_mask=attention_mask,
        )

        logits = self.output_projection(decoder_outputs.last_hidden_state)
        return logits

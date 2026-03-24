"""Qwen2/2.5-VL model forward hook for EDGE — enables Fast-V and Text-Mask attention operations."""

import torch
from typing import Optional, Union, Tuple, List

import transformers
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import BaseModelOutputWithPast
from torch.nn import CrossEntropyLoss
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast

from edge.utils.attention_mask import GetAttentionMaskwithFastVandTextMask

try:
    from transformers.masking_utils import create_causal_mask as _create_causal_mask
    _HAS_CREATE_CAUSAL_MASK = True
except ImportError:
    _HAS_CREATE_CAUSAL_MASK = False


def _compat_causal_mask(model, attention_mask, inputs_embeds, cache_position,
                        past_key_values, output_attentions, position_ids=None):
    if _HAS_CREATE_CAUSAL_MASK:
        kwargs = dict(
            config=model.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
        )
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        return _create_causal_mask(**kwargs)
    return model._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )


def _apply_2d_mask_to_4d(causal_mask_4d, augmented_mask_2d, dtype):
    if causal_mask_4d is None:
        batch, kv_len = augmented_mask_2d.shape
        mask_4d = augmented_mask_2d[:, None, None, :].to(dtype=dtype)
        mask_4d = (1.0 - mask_4d) * torch.finfo(dtype).min
        return mask_4d

    batch, mask_len = augmented_mask_2d.shape
    kv_len = causal_mask_4d.shape[-1]

    if mask_len < kv_len:
        pad = augmented_mask_2d.new_ones(batch, kv_len - mask_len)
        augmented_mask_2d = torch.cat([augmented_mask_2d, pad], dim=1)
    elif mask_len > kv_len:
        augmented_mask_2d = augmented_mask_2d[:, :kv_len]

    extra_mask = augmented_mask_2d[:, None, None, :].to(dtype=dtype, device=causal_mask_4d.device)
    extra_mask = (1.0 - extra_mask) * torch.finfo(dtype).min
    return causal_mask_4d + extra_mask


def _compat_rotary_emb(model, hidden_states, position_ids):
    try:
        return model.rotary_emb(hidden_states, position_ids=position_ids)
    except TypeError:
        return model.rotary_emb(hidden_states, position_ids)


def _is_v5_decoder_layer(decoder_layer):
    import inspect
    sig = inspect.signature(decoder_layer.forward)
    return "output_attentions" not in sig.parameters


def _run_decoder_layer_with_attn(decoder_layer, hidden_states, causal_mask,
                                 position_ids, past_key_values, output_attentions,
                                 use_cache, cache_position, position_embeddings):
    captured_attn = {}

    def _attn_hook(_module, _input, output):
        if isinstance(output, tuple) and len(output) >= 2:
            captured_attn["weights"] = output[1]

    if _is_v5_decoder_layer(decoder_layer):
        hook = None
        if output_attentions:
            hook = decoder_layer.self_attn.register_forward_hook(_attn_hook)
        try:
            out = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        finally:
            if hook is not None:
                hook.remove()

        h = out if isinstance(out, torch.Tensor) else out[0]
        attn = captured_attn.get("weights", None)
        return h, attn
    else:
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        h = layer_outputs[0]
        attn = layer_outputs[1] if output_attentions and len(layer_outputs) > 1 else None
        return h, attn


def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # EDGE additional arguments
        key_position: Optional[dict] = None,
        use_fast_v: Optional[bool] = None,
        aggregate_layer_fast_v: Optional[int] = None,
        minumum_fast_v_tokens: Optional[int] = None,
        use_text_mask: Optional[bool] = None,
        aggregate_layer_text_mask: Optional[int] = None,
        minimum_text_tokens: Optional[int] = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        use_cache = False

    if inputs_embeds is None:
        # Qwen2-VL: self.embed_tokens; Qwen2.5-VL: self.language_model.embed_tokens
        _embed_fn = getattr(self, "embed_tokens", None)
        if _embed_fn is None:
            _embed_fn = self.language_model.embed_tokens
        inputs_embeds = _embed_fn(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.dim() == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    causal_mask = _compat_causal_mask(
        self, attention_mask, inputs_embeds, cache_position,
        past_key_values, output_attentions, position_ids=position_ids
    )
    base_causal_mask = causal_mask

    hidden_states = inputs_embeds
    position_embeddings = _compat_rotary_emb(self, hidden_states, position_ids)

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    get_attention_mask_with_fast_v_and_text_mask = GetAttentionMaskwithFastVandTextMask(
        attention_mask=attention_mask,
        key_position=key_position,
        use_fast_v=use_fast_v,
        aggregate_layer_fast_v=aggregate_layer_fast_v,
        minumum_fast_v_tokens=minumum_fast_v_tokens,
        use_text_mask=use_text_mask,
        aggregate_layer_text_mask=aggregate_layer_text_mask,
        minimum_text_tokens=minimum_text_tokens,
    )

    max_attn_layer = 0
    if use_fast_v and aggregate_layer_fast_v is not None:
        max_attn_layer = max(max_attn_layer, aggregate_layer_fast_v)
    if use_text_mask and aggregate_layer_text_mask is not None:
        max_attn_layer = max(max_attn_layer, aggregate_layer_text_mask)

    for layer_idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        need_attn_this_layer = output_attentions and (layer_idx <= max_attn_layer)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states, causal_mask, position_ids, past_key_values,
                need_attn_this_layer, use_cache, cache_position, position_embeddings,
            )
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
        else:
            augmented_attention_mask = get_attention_mask_with_fast_v_and_text_mask(
                all_self_attns=all_self_attns,
            )

            if _HAS_CREATE_CAUSAL_MASK:
                causal_mask = _apply_2d_mask_to_4d(
                    base_causal_mask, augmented_attention_mask.float(), inputs_embeds.dtype
                )
            else:
                causal_mask = self._update_causal_mask(
                    augmented_attention_mask, inputs_embeds, cache_position,
                    past_key_values, need_attn_this_layer
                )

            hidden_states, attn_weights = _run_decoder_layer_with_attn(
                decoder_layer, hidden_states, causal_mask, position_ids,
                past_key_values, need_attn_this_layer, use_cache,
                cache_position, position_embeddings,
            )

            if need_attn_this_layer and attn_weights is not None:
                all_self_attns += (attn_weights,)

            if layer_idx == max_attn_layer and all_self_attns:
                all_self_attns = ()

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, past_key_values if use_cache else None, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def forward_conditional(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # EDGE additional arguments
        key_position: Optional[dict] = None,
        use_fast_v: Optional[bool] = None,
        aggregate_layer_fast_v: Optional[int] = None,
        minumum_fast_v_tokens: Optional[int] = None,
        use_text_mask: Optional[bool] = None,
        aggregate_layer_text_mask: Optional[int] = None,
        minimum_text_tokens: Optional[int] = None,
        **kwargs
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # Qwen2-VL: self.model is the language model (has embed_tokens, layers, norm)
    # Qwen2.5-VL: self.model.language_model is the language model
    _inner_model = self.model
    if not hasattr(_inner_model, "embed_tokens"):
        _inner_model = _inner_model.language_model
    _embed_tokens = _inner_model.embed_tokens

    if inputs_embeds is None:
        inputs_embeds = _embed_tokens(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.get_dtype())
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_mask = (
                (input_ids == self.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )
            video_mask = (
                (input_ids == self.config.video_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    self.rope_deltas = rope_deltas
    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        if (
            (cache_position is not None and cache_position[0] == 0)
            or self.rope_deltas is None
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        ):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, attention_mask
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                delta = delta.to(position_ids.device)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = _inner_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        key_position=key_position,
        use_fast_v=use_fast_v,
        aggregate_layer_fast_v=aggregate_layer_fast_v,
        minumum_fast_v_tokens=minumum_fast_v_tokens,
        use_text_mask=use_text_mask,
        aggregate_layer_text_mask=aggregate_layer_text_mask,
        minimum_text_tokens=minimum_text_tokens,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        logits = logits.float()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )


def patch_qwen_forward():
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel.forward = forward
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.forward = forward_conditional

    try:
        from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
        # Qwen2.5-VL nests layers/norm/embed_tokens inside language_model (Qwen2_5_VLTextModel),
        # so patch the text model, not Qwen2_5_VLModel itself.
        modeling_qwen2_5_vl.Qwen2_5_VLTextModel.forward = forward
        modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = forward_conditional
    except (ImportError, AttributeError):
        pass

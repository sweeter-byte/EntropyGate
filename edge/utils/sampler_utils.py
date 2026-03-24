"""Utility functions for EDGE sampler forward passes."""


def get_generations(self,
                    input_ids,
                    pixel_values,
                    model_kwargs,
                    generation_config,
                    key_position,
                    output_attentions,
                    use_text_mask,
                    use_fast_v,
                    output_hidden_states):

    # Detect first iteration: no KV cache yet
    past_kv = model_kwargs.get("past_key_values", None)
    is_first = past_kv is None or (hasattr(past_kv, "get_seq_length") and past_kv.get_seq_length() == 0)

    # Autoregressive generation: only feed the last generated token if not first iter
    if not is_first:
        input_ids = input_ids[:, -1:]
        pixel_values = None

    model_inputs = self.prepare_inputs_for_generation(input_ids=input_ids,
                                                      pixel_values=pixel_values,
                                                      is_first_iteration=is_first,
                                                      **model_kwargs)

    rope_deltas = model_kwargs.get("rope_deltas", None)
    model_inputs.update({"rope_deltas": rope_deltas} if rope_deltas is not None else {})
    model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

    # Only compute attention weights for passes that need attention masks
    effective_output_attentions = True if (use_fast_v or use_text_mask) else output_attentions
    model_inputs["output_attentions"] = effective_output_attentions

    outputs = self(
        **model_inputs,
        return_dict=True,
        key_position=key_position,
        use_fast_v=use_fast_v,
        aggregate_layer_fast_v=generation_config.aggregate_layer_fast_v,
        minumum_fast_v_tokens=generation_config.minumum_fast_v_tokens,
        use_text_mask=use_text_mask,
        aggregate_layer_text_mask=generation_config.aggregate_layer_text_mask,
        minimum_text_tokens=generation_config.minimum_text_tokens,
    )

    if ('rope_deltas' in outputs) and ('rope_deltas' not in model_kwargs):
        model_kwargs['rope_deltas'] = outputs['rope_deltas']

    # Compatible with multiple transformers API versions
    try:
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder,
        )
    except TypeError:
        try:
            model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs)
        except TypeError:
            if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
                model_kwargs["past_key_values"] = outputs.past_key_values
            if "cache_position" in model_kwargs:
                model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1

    return outputs, model_kwargs


def get_next_token_logits(outputs, input_ids):
    next_token_logits = outputs.logits[:, -1, :].clone().float()
    next_token_logits = next_token_logits.to(input_ids.device)
    return next_token_logits

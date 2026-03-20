"""VCD sampling: Visual Contrastive Decoding for hallucination mitigation.

Supports two modes controlled by `vcd_entropy_gate`:
  - False (default): original VCD with fixed cd_alpha
  - True: VCD + EntropyGate with entropy-gated dynamic alpha

Ported to transformers >=4.48 API, following the same conventions as
crops_sample.py and entropygate_sample.py in this project.
"""
import math
import logging
from typing import Optional, Union

import torch
from torch import nn

from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
import transformers
from transformers.generation.utils import (
    GenerateNonBeamOutput,
    GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput,
)
from transformers.generation.streamers import BaseStreamer

from methods.generation_configs.vcd_generation_config import GenerationConfigVCD
from methods.samplers.entropygate_sample import (
    _compat_get_initial_cache_position,
    _compat_has_unfinished,
)

logger = logging.getLogger("entropygate")


def vcd_sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfigVCD,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    """VCD sampling with optional EntropyGate."""
    # init values
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    max_length = generation_config.max_length
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    #### VCD parameters ####
    cd_alpha = generation_config.vcd_alpha
    cd_beta = torch.tensor(generation_config.vcd_beta)

    # EntropyGate parameters (only used when vcd_entropy_gate=True)
    vcd_entropy_gate = getattr(generation_config, "vcd_entropy_gate", False)
    eg_alpha_min = getattr(generation_config, "eg_alpha_min", 0.5)
    eg_alpha_max = getattr(generation_config, "eg_alpha_max", 1.5)
    eg_eta = getattr(generation_config, "eg_eta", 0.10)
    eg_tau = getattr(generation_config, "eg_tau", 0.05)

    # pixel_values for original image (kept in model_kwargs by caller)
    # pixel_values_cd for noised image (injected via generation_config)
    pixel_values = model_kwargs.pop("pixel_values", None)
    pixel_values_cd = getattr(generation_config, "pixel_values_cd", None)

    # Separate model_kwargs for the contrastive (noised) branch
    import copy
    model_kwargs_cd = copy.deepcopy(model_kwargs)

    # Logging accumulators
    _log_H_vals = []
    _log_alpha_vals = []
    _log_total = 0

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

    model_kwargs = _compat_get_initial_cache_position(self, input_ids, model_kwargs)
    model_kwargs_cd = _compat_get_initial_cache_position(self, input_ids, model_kwargs_cd)

    # PLACEHOLDER_LOOP

    while _compat_has_unfinished(self, this_peer_finished, synced_gpus, input_ids.device, cur_len, max_length):

        # ---- Forward pass 1: original model (clean image) ----
        pv = pixel_values if cur_len == input_ids.shape[1] else None
        # Detect first iteration
        past_kv = model_kwargs.get("past_key_values", None)
        is_first = past_kv is None or (hasattr(past_kv, "get_seq_length") and past_kv.get_seq_length() == 0)
        feed_ids = input_ids if is_first else input_ids[:, -1:]
        feed_pv = pixel_values if is_first else None

        model_inputs = self.prepare_inputs_for_generation(
            input_ids=feed_ids, pixel_values=feed_pv,
            is_first_iteration=is_first, **model_kwargs
        )
        rope_deltas = model_kwargs.get("rope_deltas", None)
        if rope_deltas is not None:
            model_inputs["rope_deltas"] = rope_deltas
        model_inputs["output_attentions"] = False

        outputs = self(**model_inputs, return_dict=True)

        if "rope_deltas" in outputs and "rope_deltas" not in model_kwargs:
            model_kwargs["rope_deltas"] = outputs["rope_deltas"]

        # Update model_kwargs for next step
        try:
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder)
        except TypeError:
            try:
                model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs)
            except TypeError:
                if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
                    model_kwargs["past_key_values"] = outputs.past_key_values
                if "cache_position" in model_kwargs:
                    model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1

        if synced_gpus and this_peer_finished:
            continue

        next_token_logits = outputs.logits[:, -1, :].clone().float().to(input_ids.device)
        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---- Forward pass 2: contrastive model (noised image) ----
        past_kv_cd = model_kwargs_cd.get("past_key_values", None)
        is_first_cd = past_kv_cd is None or (hasattr(past_kv_cd, "get_seq_length") and past_kv_cd.get_seq_length() == 0)
        feed_ids_cd = input_ids if is_first_cd else input_ids[:, -1:]
        feed_pv_cd = pixel_values_cd if is_first_cd else None

        model_inputs_cd = self.prepare_inputs_for_generation(
            input_ids=feed_ids_cd, pixel_values=feed_pv_cd,
            is_first_iteration=is_first_cd, **model_kwargs_cd
        )
        rope_deltas_cd = model_kwargs_cd.get("rope_deltas", None)
        if rope_deltas_cd is not None:
            model_inputs_cd["rope_deltas"] = rope_deltas_cd
        model_inputs_cd["output_attentions"] = False

        outputs_cd = self(**model_inputs_cd, return_dict=True)

        if "rope_deltas" in outputs_cd and "rope_deltas" not in model_kwargs_cd:
            model_kwargs_cd["rope_deltas"] = outputs_cd["rope_deltas"]

        try:
            model_kwargs_cd = self._update_model_kwargs_for_generation(
                outputs_cd, model_kwargs_cd, is_encoder_decoder=self.config.is_encoder_decoder)
        except TypeError:
            try:
                model_kwargs_cd = self._update_model_kwargs_for_generation(outputs_cd, model_kwargs_cd)
            except TypeError:
                if hasattr(outputs_cd, "past_key_values") and outputs_cd.past_key_values is not None:
                    model_kwargs_cd["past_key_values"] = outputs_cd.past_key_values
                if "cache_position" in model_kwargs_cd:
                    model_kwargs_cd["cache_position"] = model_kwargs_cd["cache_position"][-1:] + 1

        next_token_logits_cd = outputs_cd.logits[:, -1, :].clone().float().to(input_ids.device)
        del outputs_cd
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # PLACEHOLDER_CONTRASTIVE

        # ==== VCD contrastive decoding ====
        _log_total += 1

        # Determine alpha: fixed or entropy-gated
        if vcd_entropy_gate:
            probs_orig = torch.softmax(next_token_logits, dim=-1)
            vocab_size = next_token_logits.shape[-1]
            H_t = -(probs_orig * torch.log(probs_orig + 1e-12)).sum(dim=-1, keepdim=True) / math.log(vocab_size)
            alpha_t = eg_alpha_min + (eg_alpha_max - eg_alpha_min) * torch.sigmoid((H_t - eg_eta) / eg_tau)
            alpha_val = alpha_t.item()
            _log_H_vals.append(H_t.item())
            _log_alpha_vals.append(alpha_val)
        else:
            alpha_val = cd_alpha
            alpha_t = cd_alpha

        # Adaptive plausibility constraint (cutoff)
        cutoff = torch.log(cd_beta) + next_token_logits.max(dim=-1, keepdim=True).values

        # Contrastive logits
        diffs = (1 + alpha_t) * next_token_logits - alpha_t * next_token_logits_cd
        final_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, final_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (final_logits,)

        # token selection
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

    if streamer is not None:
        streamer.end()

    # Per-sample summary log
    if _log_total > 0 and vcd_entropy_gate and _log_H_vals:
        avg_H = sum(_log_H_vals) / len(_log_H_vals)
        avg_alpha = sum(_log_alpha_vals) / len(_log_alpha_vals)
        logger.info(
            f"[vcd_eg_summary] steps={_log_total} "
            f"avg_H={avg_H:.4f} avg_alpha={avg_alpha:.4f}"
        )

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids, scores=scores, logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids, scores=scores, logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids


def patch_vcd_sampling():
    transformers.generation.utils.GenerationMixin._sample = vcd_sample

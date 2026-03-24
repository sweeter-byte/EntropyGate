"""EDGE sampler: entropy-gated nested contrastive decoding.

Core formula (E5):
    intermediate = log_p + (1 - gamma_t) / gamma_t * (log_p - log_p_lang)
    final = (1 + g_vis) * intermediate - g_vis * log_p_stat
    g_vis = alpha_min + (alpha_base - alpha_min) * sigmoid((H_t - eta) / tau)
"""

import math
import copy
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

from edge.generation_config import EdgeGenerationConfig
from edge.utils.sampler_utils import get_generations, get_next_token_logits

logger = logging.getLogger("edge")


def _compat_get_initial_cache_position(model, input_ids, model_kwargs):
    """Call _get_initial_cache_position with API-version compatibility."""
    if not hasattr(model, "_get_initial_cache_position"):
        model_kwargs["cache_position"] = torch.arange(
            0, input_ids.shape[-1], device=input_ids.device
        )
        return model_kwargs
    try:
        return model._get_initial_cache_position(input_ids.shape[-1], input_ids.device, model_kwargs)
    except TypeError:
        pass
    try:
        return model._get_initial_cache_position(input_ids, model_kwargs)
    except TypeError:
        pass
    model_kwargs["cache_position"] = torch.arange(
        0, input_ids.shape[-1], device=input_ids.device
    )
    return model_kwargs


def _compat_has_unfinished(model, this_peer_finished, synced_gpus, device, cur_len, max_length):
    """Call _has_unfinished_sequences with API-version compatibility."""
    if not hasattr(model, "_has_unfinished_sequences"):
        return not this_peer_finished
    try:
        return model._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=device, cur_len=cur_len, max_length=max_length
        )
    except TypeError:
        pass
    try:
        return model._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=device
        )
    except TypeError:
        pass
    return not this_peer_finished


def _new_text_tokens(t, b0=10, b1=30, lamb=0.001):
    return math.floor(b0 + b1 * (1 - math.exp(-lamb * t)))


def edge_sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: EdgeGenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    """EDGE sampling: entropy-gated nested contrastive decoding for hallucination mitigation."""

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

    # ---- Initialise EDGE variables ----
    pixel_values = model_kwargs.pop("pixel_values", None)
    key_position = generation_config.key_position

    # Lang Prior (text-visual joint hallucination model)
    model_kwargs_lang_prior = copy.deepcopy(model_kwargs)
    input_ids_lang_prior = generation_config.input_ids_lang_prior
    model_kwargs_lang_prior["attention_mask"] = torch.ones_like(input_ids_lang_prior)

    # Stat Bias (visual hallucination model)
    model_kwargs_stat_bias = copy.deepcopy(model_kwargs)

    # EDGE hyperparameters
    alpha_base_vis = generation_config.alpha_base_vis
    alpha_min_vis = generation_config.alpha_min_vis
    eta_vis = generation_config.eta_vis
    tau_gate = generation_config.tau_gate
    gamma_decay = generation_config.gamma_decay
    beta_cutoff = generation_config.beta_cutoff
    theta_safe = generation_config.theta_safe

    time_step = 1

    # Logging accumulators
    _log_H_vals = []
    _log_g_vis_vals = []
    _log_skip_count = 0
    _log_gate_count = 0

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

    model_kwargs = _compat_get_initial_cache_position(self, input_ids, model_kwargs)
    model_kwargs_lang_prior = _compat_get_initial_cache_position(self, input_ids_lang_prior, model_kwargs_lang_prior)
    model_kwargs_stat_bias = _compat_get_initial_cache_position(self, input_ids, model_kwargs_stat_bias)

    while _compat_has_unfinished(self, this_peer_finished, synced_gpus, input_ids.device, cur_len, max_length):

        # ---- Forward pass 1: original model ----
        outputs, model_kwargs = get_generations(self,
                                                input_ids,
                                                pixel_values=pixel_values,
                                                model_kwargs=model_kwargs,
                                                generation_config=generation_config,
                                                key_position=None,
                                                use_text_mask=False,
                                                use_fast_v=False,
                                                output_attentions=False,
                                                output_hidden_states=output_hidden_states)

        if synced_gpus and this_peer_finished:
            continue

        next_token_logits = get_next_token_logits(outputs, input_ids)
        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        generation_config.minimum_text_tokens = _new_text_tokens(time_step)

        # ---- Forward pass 2: lang prior ----
        outputs_lang_prior, model_kwargs_lang_prior = get_generations(self,
                                                input_ids_lang_prior,
                                                pixel_values=None,
                                                model_kwargs=model_kwargs_lang_prior,
                                                generation_config=generation_config,
                                                key_position=key_position,
                                                use_text_mask=True,
                                                use_fast_v=False,
                                                output_attentions=output_attentions,
                                                output_hidden_states=output_hidden_states)

        if synced_gpus and this_peer_finished:
            continue

        next_token_logits_lang_prior = get_next_token_logits(outputs_lang_prior, input_ids_lang_prior)
        del outputs_lang_prior
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---- Forward pass 3: stat bias ----
        outputs_stat_bias, model_kwargs_stat_bias = get_generations(self,
                                                input_ids,
                                                pixel_values=pixel_values,
                                                model_kwargs=model_kwargs_stat_bias,
                                                generation_config=generation_config,
                                                key_position=key_position,
                                                use_text_mask=False,
                                                use_fast_v=True,
                                                output_attentions=output_attentions,
                                                output_hidden_states=output_hidden_states)

        if synced_gpus and this_peer_finished:
            continue

        next_token_logits_stat_bias = get_next_token_logits(outputs_stat_bias, input_ids)
        del outputs_stat_bias
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ==== EDGE core logic ====

        # Step 1: Compute normalized entropy H_t from original logits
        probs_orig = torch.softmax(next_token_logits, dim=-1)
        vocab_size = next_token_logits.shape[-1]
        H_t = -(probs_orig * torch.log(probs_orig + 1e-12)).sum(dim=-1, keepdim=True) / math.log(vocab_size)
        max_prob = probs_orig.max(dim=-1, keepdim=True).values.item()

        # Step 2: Safety skip — when model is very confident, skip contrastive
        if max_prob > theta_safe:
            final_logits = next_token_logits
            _log_skip_count += 1
            _log_H_vals.append(H_t.item())
            _log_g_vis_vals.append(0.0)
            logger.debug(f"step={time_step:03d} H_t={H_t.item():.4f} SKIP (max_prob>{theta_safe:.2f})")
        else:
            # ---- Plausibility cutoff ----
            cutoff_th = math.log(beta_cutoff) + next_token_logits.max(dim=-1, keepdim=True).values
            next_token_logits = next_token_logits.masked_fill(next_token_logits < cutoff_th, -float("inf"))

            # ---- Log-softmax of all three distributions ----
            log_probs = torch.log_softmax(next_token_logits, dim=-1)
            log_probs_lang = torch.log_softmax(next_token_logits_lang_prior, dim=-1)
            log_probs_stat = torch.log_softmax(next_token_logits_stat_bias, dim=-1)

            # ---- Nested contrastive with entropy gate ----
            gamma_time = math.exp(-gamma_decay * time_step)
            time_mult = (1 - gamma_time) / (gamma_time + 1e-12)

            # Step 1: Language prior contrast (CRoPS time-decayed)
            intermediate = log_probs + time_mult * (log_probs - log_probs_lang)

            # Step 2: Entropy-gated visual contrast
            gate_frac_vis = torch.sigmoid((H_t - eta_vis) / tau_gate)
            g_vis = alpha_min_vis + (alpha_base_vis - alpha_min_vis) * gate_frac_vis

            # Step 3: Final nested combination
            final_logits = (1 + g_vis) * intermediate - g_vis * log_probs_stat

            _log_gate_count += 1
            _log_H_vals.append(H_t.item())
            _log_g_vis_vals.append(g_vis.item() if torch.is_tensor(g_vis) else g_vis)
            logger.debug(
                f"step={time_step:03d} H_t={H_t.item():.4f} "
                f"g_vis={g_vis.item() if torch.is_tensor(g_vis) else g_vis:.4f} "
                f"time_mult={time_mult:.4f}"
            )

        time_step += 1

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, final_logits)

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

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        input_ids_lang_prior = torch.cat([input_ids_lang_prior, next_tokens[:, None]], dim=-1)

        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

    if streamer is not None:
        streamer.end()

    # Per-sample summary log
    total_steps = _log_skip_count + _log_gate_count
    if total_steps > 0 and _log_H_vals:
        avg_H = sum(_log_H_vals) / len(_log_H_vals)
        avg_g_vis = sum(_log_g_vis_vals) / len(_log_g_vis_vals)
        skip_rate = _log_skip_count / total_steps
        logger.info(
            f"[sample_summary] steps={total_steps} "
            f"avg_H={avg_H:.4f} avg_g_vis={avg_g_vis:.4f} "
            f"skip_rate={skip_rate:.2%} ({_log_skip_count}/{total_steps})"
        )

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids


def patch_edge_sampling():
    """Monkey-patch transformers to use EDGE sampling."""
    transformers.generation.utils.GenerationMixin._sample = edge_sample

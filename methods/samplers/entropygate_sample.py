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

from methods.generation_configs.entropygate_generation_config import GenerationConfigEntropyGate
from methods.utils.crops_samplers_utils import get_generations, get_next_token_logits

logger = logging.getLogger("entropygate")


def _compat_get_initial_cache_position(model, input_ids, model_kwargs):
    """Call _get_initial_cache_position with API-version compatibility."""
    if not hasattr(model, "_get_initial_cache_position"):
        # Very old transformers — just set cache_position manually
        model_kwargs["cache_position"] = torch.arange(
            0, input_ids.shape[-1], device=input_ids.device
        )
        return model_kwargs
    # transformers 5.x: _get_initial_cache_position(seq_length: int, device, model_kwargs)
    try:
        return model._get_initial_cache_position(input_ids.shape[-1], input_ids.device, model_kwargs)
    except TypeError:
        pass
    # transformers 4.x: _get_initial_cache_position(input_ids, model_kwargs)
    try:
        return model._get_initial_cache_position(input_ids, model_kwargs)
    except TypeError:
        pass
    # Fallback
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


def _compat_update_model_kwargs(model, outputs, model_kwargs, is_encoder_decoder):
    """Call _update_model_kwargs_for_generation with API-version compatibility."""
    try:
        return model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=is_encoder_decoder
        )
    except TypeError:
        pass
    try:
        return model._update_model_kwargs_for_generation(outputs, model_kwargs)
    except TypeError:
        pass
    # Minimal fallback: update past_key_values and cache_position
    if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
        model_kwargs["past_key_values"] = outputs.past_key_values
    if "cache_position" in model_kwargs:
        model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1
    return model_kwargs


def entropygate_sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfigEntropyGate,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    r"""
    EntropyGate sampling: entropy-gated contrastive decoding for hallucination mitigation.
    """
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

    #### Initialise variables for EntropyGate ####
    pixel_values = model_kwargs.pop("pixel_values", None)
    key_position = generation_config.key_position

    # Lang Prior (text-visual joint hallucination model)
    model_kwargs_lang_prior = copy.deepcopy(model_kwargs)
    input_ids_lang_prior = generation_config.input_ids_lang_prior
    model_kwargs_lang_prior["attention_mask"] = torch.ones_like(input_ids_lang_prior)

    # Stat Bias (visual hallucination model)
    model_kwargs_stat_bias = copy.deepcopy(model_kwargs)

    # EntropyGate hyperparameters
    alpha_base_vis = generation_config.alpha_base_vis
    alpha_base_txt = generation_config.alpha_base_txt
    alpha_min_vis = generation_config.alpha_min_vis
    alpha_min_txt = generation_config.alpha_min_txt
    eta_vis = generation_config.eta_vis
    eta_txt = generation_config.eta_txt
    tau_gate = generation_config.tau_gate
    gamma_decay = generation_config.gamma_decay
    beta_base = generation_config.beta_base
    beta_range = generation_config.beta_range
    theta_safe = torch.tensor(generation_config.theta_safe, device=input_ids.device)
    eg_scheme = generation_config.eg_scheme
    beta_cutoff_fixed = generation_config.beta_cutoff_fixed
    theta_safe_aligned = generation_config.theta_safe_aligned
    time_decay_mode = generation_config.time_decay_mode
    alpha_time_txt = generation_config.alpha_time_txt
    adaptive_eta = generation_config.adaptive_eta
    eta_ema_momentum = generation_config.eta_ema_momentum
    eta_vis_offset = generation_config.eta_vis_offset
    eta_txt_offset = generation_config.eta_txt_offset
    soft_suppress = generation_config.soft_suppress
    soft_suppress_k = generation_config.soft_suppress_k

    # Direction 3: adaptive eta running stats
    _ema_H_mean = 0.12  # warm-start near typical value
    _ema_H_var = 0.001

    time_step = 1

    # Logging accumulators
    _log_H_vals = []
    _log_g_vis_vals = []
    _log_g_txt_vals = []
    _log_skip_count = 0
    _log_gate_count = 0

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

    model_kwargs = _compat_get_initial_cache_position(self, input_ids, model_kwargs)
    model_kwargs_lang_prior = _compat_get_initial_cache_position(self, input_ids_lang_prior, model_kwargs_lang_prior)
    model_kwargs_stat_bias = _compat_get_initial_cache_position(self, input_ids, model_kwargs_stat_bias)

    # PLACEHOLDER_MAIN_LOOP

    while _compat_has_unfinished(self, this_peer_finished, synced_gpus, input_ids.device, cur_len, max_length):

        # ---- Forward pass 1: original model ----
        # Note: output_attentions=False here because pass 1 doesn't use attention weights,
        # saving significant GPU memory (~1.5 GiB for 32 layers).
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

        generation_config.minimum_text_tokens = new_text_tokens(time_step)

        # ---- Forward pass 2: lang prior (text-visual joint hallucination) ----
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

        # ---- Forward pass 3: stat bias (visual hallucination) ----
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

        # ==== EntropyGate core logic ====

        # Step 1: Compute normalized entropy H_t from original logits
        probs_orig = torch.softmax(next_token_logits, dim=-1)
        vocab_size = next_token_logits.shape[-1]
        H_t = -(probs_orig * torch.log(probs_orig + 1e-12)).sum(dim=-1, keepdim=True) / math.log(vocab_size)
        max_prob = probs_orig.max(dim=-1, keepdim=True).values.item()

        # Direction 3: update adaptive eta EMA
        if adaptive_eta:
            h_val = H_t.item()
            _ema_H_mean = (1 - eta_ema_momentum) * _ema_H_mean + eta_ema_momentum * h_val
            _ema_H_var = (1 - eta_ema_momentum) * _ema_H_var + eta_ema_momentum * (h_val - _ema_H_mean) ** 2

        # Step 2: Safety skip / soft suppress
        effective_theta = theta_safe_aligned if eg_scheme == "nested_aligned" else theta_safe.item()

        if not soft_suppress and max_prob > effective_theta:
            # Hard skip
            final_logits = next_token_logits
            _log_skip_count += 1
            _log_H_vals.append(H_t.item())
            _log_g_vis_vals.append(0.0)
            _log_g_txt_vals.append(0.0)
            logger.debug(f"step={time_step:03d} H_t={H_t.item():.4f} SKIP (max_prob>{effective_theta:.2f})")
        else:
            # ---- Cutoff ----
            if eg_scheme in ("nested", "nested_aligned", "acd"):
                cutoff_th = math.log(beta_cutoff_fixed) + next_token_logits.max(dim=-1, keepdim=True).values
            else:
                beta_t = beta_base + beta_range * (1 - H_t)
                cutoff_th = torch.log(beta_t) + next_token_logits.max(dim=-1, keepdim=True).values
            next_token_logits = next_token_logits.masked_fill(next_token_logits < cutoff_th, -float("inf"))

            # ---- Log-softmax of all three distributions ----
            log_probs = torch.log_softmax(next_token_logits, dim=-1)
            log_probs_lang = torch.log_softmax(next_token_logits_lang_prior, dim=-1)
            log_probs_stat = torch.log_softmax(next_token_logits_stat_bias, dim=-1)

            # ---- Time decay ----
            gamma_time = math.exp(-gamma_decay * time_step)

            if eg_scheme == "flat":
                # === Original flat formula with direction 1/2/3/4 support ===

                # Direction 3: adaptive eta
                if adaptive_eta:
                    h_std = max(math.sqrt(_ema_H_var), 1e-6)
                    eff_eta_vis = _ema_H_mean + eta_vis_offset * h_std
                    eff_eta_txt = _ema_H_mean + eta_txt_offset * h_std
                else:
                    eff_eta_vis = eta_vis
                    eff_eta_txt = eta_txt

                # Entropy gates
                gate_frac_vis = torch.sigmoid((H_t - eff_eta_vis) / tau_gate)
                g_vis = alpha_min_vis + (alpha_base_vis - alpha_min_vis) * gate_frac_vis

                gate_frac_txt = torch.sigmoid((H_t - eff_eta_txt) / tau_gate)
                # Direction 2: time decay mode
                if time_decay_mode == "additive":
                    g_txt_base = alpha_min_txt + (alpha_base_txt - alpha_min_txt) * gate_frac_txt
                    time_bonus = alpha_time_txt * (1 - gamma_time) / (gamma_time + 1e-12)
                    g_txt = g_txt_base + time_bonus
                else:
                    g_txt = (alpha_min_txt + (alpha_base_txt - alpha_min_txt) * gate_frac_txt) \
                            * (1 - gamma_time) / (gamma_time + 1e-12)

                # Direction 4: soft suppress
                if soft_suppress:
                    suppress_factor = 1.0 - max_prob ** soft_suppress_k
                    g_vis = g_vis * suppress_factor
                    g_txt = g_txt * suppress_factor

                final_logits = (1 + g_vis + g_txt) * log_probs \
                               - g_vis * log_probs_stat \
                               - g_txt * log_probs_lang

            elif eg_scheme in ("nested", "nested_aligned"):
                time_mult = (1 - gamma_time) / (gamma_time + 1e-12)
                intermediate = log_probs + time_mult * (log_probs - log_probs_lang)
                gate_frac_vis = torch.sigmoid((H_t - eta_vis) / tau_gate)
                g_vis = alpha_min_vis + (alpha_base_vis - alpha_min_vis) * gate_frac_vis
                final_logits = (1 + g_vis) * intermediate - g_vis * log_probs_stat
                g_txt = torch.tensor(time_mult)

            elif eg_scheme == "acd":
                probs_stat = torch.softmax(next_token_logits_stat_bias, dim=-1)
                probs_lang = torch.softmax(next_token_logits_lang_prior, dim=-1)
                H_stat = -(probs_stat * torch.log(probs_stat + 1e-12)).sum(dim=-1, keepdim=True) / math.log(vocab_size)
                H_lang = -(probs_lang * torch.log(probs_lang + 1e-12)).sum(dim=-1, keepdim=True) / math.log(vocab_size)
                alpha_vis_acd = H_t / (H_t + H_stat + 1e-12)
                alpha_txt_acd = H_t / (H_t + H_lang + 1e-12)
                g_vis = torch.max(alpha_vis_acd, torch.tensor(alpha_min_vis, device=H_t.device))
                g_txt_base = torch.max(alpha_txt_acd, torch.tensor(alpha_min_txt, device=H_t.device))
                time_mult = (1 - gamma_time) / (gamma_time + 1e-12)
                intermediate = log_probs + g_txt_base * time_mult * (log_probs - log_probs_lang)
                final_logits = (1 + g_vis) * intermediate - g_vis * log_probs_stat
                g_txt = g_txt_base * time_mult

            _log_gate_count += 1
            _log_H_vals.append(H_t.item())
            _log_g_vis_vals.append(g_vis.item() if torch.is_tensor(g_vis) else g_vis)
            _log_g_txt_vals.append(g_txt.item() if torch.is_tensor(g_txt) else g_txt)
            logger.debug(
                f"step={time_step:03d} H_t={H_t.item():.4f} "
                f"g_vis={g_vis.item() if torch.is_tensor(g_vis) else g_vis:.4f} "
                f"g_txt={g_txt.item() if torch.is_tensor(g_txt) else g_txt:.4f} "
                f"scheme={eg_scheme}"
            )

        time_step += 1

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, final_logits)

        # Store scores, attentions and hidden_states when required
        # Note: outputs was deleted after extracting logits to save GPU memory,
        # so per-step attentions/hidden_states are not available.
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

        # outputs already deleted above after extracting logits

    if streamer is not None:
        streamer.end()

    # Per-sample summary log
    total_steps = _log_skip_count + _log_gate_count
    if total_steps > 0 and _log_H_vals:
        avg_H = sum(_log_H_vals) / len(_log_H_vals)
        avg_g_vis = sum(_log_g_vis_vals) / len(_log_g_vis_vals)
        avg_g_txt = sum(_log_g_txt_vals) / len(_log_g_txt_vals)
        skip_rate = _log_skip_count / total_steps
        logger.info(
            f"[sample_summary] steps={total_steps} "
            f"avg_H={avg_H:.4f} avg_g_vis={avg_g_vis:.4f} avg_g_txt={avg_g_txt:.4f} "
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


def patch_entropygate_sampling():
    transformers.generation.utils.GenerationMixin._sample = entropygate_sample


def new_text_tokens(t, b0=10, b1=30, lamb=0.001):
    return math.floor(b0 + b1 * (1 - math.exp(-lamb * t)))

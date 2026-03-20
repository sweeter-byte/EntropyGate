"""Latent-space sampling methods for hallucination mitigation.

Three methods are implemented in a unified sampler:

HSC (Hidden State Contrastive):
    Stat-bias contrast happens in hidden-state space (pre-norm), then
    norm + lm_head projects back to logits.  Lang prior contrast is in
    logit space (nested structure with time decay).  RMSNorm after contrast
    introduces non-linearity, making this different from logit-space contrast.

LEG (Latent Entropy Gate):
    Everything stays in logit space (same nested formula as EntropyGate E5),
    but the entropy signal that drives the gate is computed from a MIDDLE
    hidden layer (projected via norm + lm_head) instead of output logits.
    Mid-layer entropy may detect hallucination risk earlier.

LLH (Latent-Logit Hybrid):
    Two stages — (1) stat-bias contrast in hidden space (entropy-gated,
    like HSC), then (2) lang-prior contrast in logit space with an
    additional entropy gate on the corrected distribution (not just time
    decay).  Double entropy gating.
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

from methods.generation_configs.latent_generation_config import GenerationConfigLatent
from methods.utils.crops_samplers_utils import get_generations, get_next_token_logits
from methods.utils.latent_utils import (
    get_hidden_state_at_layer,
    project_prenorm_to_logits,
)
from methods.samplers.entropygate_sample import (
    _compat_get_initial_cache_position,
    _compat_has_unfinished,
    new_text_tokens,
)

logger = logging.getLogger("entropygate")


# ---------------------------------------------------------------------------
# Core logic helpers for each method
# ---------------------------------------------------------------------------

def _hsc_core(model, next_token_logits, next_token_logits_lang,
              h_orig, h_stat, H_t,
              alpha_base, alpha_min, eta, tau,
              gamma_decay, beta_cutoff, time_step):
    """HSC: Hidden State Contrastive.

    Stage 1: Entropy-gated contrast in hidden space (stat bias).
        h_contrasted = h_orig + g_vis * (h_orig - h_stat)
        logits_corrected = lm_head(norm(h_contrasted))
    Stage 2: Lang prior contrast in logit space (nested, time decay).
        final = log_p_corrected + time_mult * (log_p_corrected - log_p_lang)
    """
    # Entropy gate
    gate_frac = torch.sigmoid((H_t - eta) / tau)
    g_vis = alpha_min + (alpha_base - alpha_min) * gate_frac

    # Hidden-space contrast
    h_contrasted = h_orig + g_vis * (h_orig - h_stat)

    # Project to logits via norm (non-linear!) + lm_head
    logits_corrected = project_prenorm_to_logits(model, h_contrasted)

    # Plausibility cutoff on corrected logits
    cutoff_th = math.log(beta_cutoff) + logits_corrected.max(dim=-1, keepdim=True).values
    logits_corrected = logits_corrected.masked_fill(logits_corrected < cutoff_th, -float("inf"))

    # Lang prior contrast in logit space (nested structure)
    gamma_time = math.exp(-gamma_decay * time_step)
    time_mult = (1 - gamma_time) / (gamma_time + 1e-12)

    log_p_corrected = torch.log_softmax(logits_corrected, dim=-1)
    log_p_lang = torch.log_softmax(next_token_logits_lang, dim=-1)

    final = log_p_corrected + time_mult * (log_p_corrected - log_p_lang)

    return final, g_vis, time_mult


def _leg_core(model, next_token_logits, next_token_logits_lang,
              next_token_logits_stat, h_layer,
              alpha_base, alpha_min, eta, tau,
              gamma_decay, beta_cutoff, time_step):
    """LEG: Latent Entropy Gate.

    Compute entropy from a middle hidden layer's projected logits (instead
    of output logits).  All contrast is in logit space (nested formula).
    """
    # Compute entropy from hidden layer projected logits
    logits_from_h = project_prenorm_to_logits(model, h_layer)
    probs_h = torch.softmax(logits_from_h, dim=-1)
    vocab_size = logits_from_h.shape[-1]
    H_latent = -(probs_h * torch.log(probs_h + 1e-12)).sum(dim=-1, keepdim=True) / math.log(vocab_size)

    # Entropy gate driven by LATENT entropy
    gate_frac = torch.sigmoid((H_latent - eta) / tau)
    g_vis = alpha_min + (alpha_base - alpha_min) * gate_frac

    # Plausibility cutoff on original logits
    cutoff_th = math.log(beta_cutoff) + next_token_logits.max(dim=-1, keepdim=True).values
    logits_cut = next_token_logits.masked_fill(next_token_logits < cutoff_th, -float("inf"))

    # Nested formula in logit space (same as EntropyGate E5)
    gamma_time = math.exp(-gamma_decay * time_step)
    time_mult = (1 - gamma_time) / (gamma_time + 1e-12)

    log_p = torch.log_softmax(logits_cut, dim=-1)
    log_p_lang = torch.log_softmax(next_token_logits_lang, dim=-1)
    log_p_stat = torch.log_softmax(next_token_logits_stat, dim=-1)

    intermediate = log_p + time_mult * (log_p - log_p_lang)
    final = (1 + g_vis) * intermediate - g_vis * log_p_stat

    return final, g_vis, time_mult


def _llh_core(model, next_token_logits, next_token_logits_lang,
              h_orig, h_stat, H_t,
              hidden_alpha_base, hidden_alpha_min, hidden_eta, hidden_tau,
              gamma_decay, beta_cutoff, time_step):
    """LLH: Latent-Logit Hybrid.

    Stage 1: Entropy-gated hidden-space contrast (stat bias) — same as HSC.
    Stage 2: Entropy-gated logit-space contrast (lang prior) —
        uses entropy of the CORRECTED distribution to gate lang prior
        strength, not just time decay.  This gives double entropy gating.

    Formula:
        h_contrasted = h_orig + g_vis_h * (h_orig - h_stat)
        logits_corrected = lm_head(norm(h_contrasted))
        H_corrected = entropy(logits_corrected)
        g_txt = time_mult * sigmoid((H_corrected - eta) / tau)
        final = (1 + g_txt) * log_p_corrected - g_txt * log_p_lang
    """
    # Stage 1: Hidden-space stat-bias contrast
    gate_frac_h = torch.sigmoid((H_t - hidden_eta) / hidden_tau)
    g_vis = hidden_alpha_min + (hidden_alpha_base - hidden_alpha_min) * gate_frac_h

    h_contrasted = h_orig + g_vis * (h_orig - h_stat)
    logits_corrected = project_prenorm_to_logits(model, h_contrasted)

    # Stage 2: Entropy-gated lang prior contrast
    # Compute entropy from CORRECTED logits
    probs_corrected = torch.softmax(logits_corrected, dim=-1)
    vocab_size = logits_corrected.shape[-1]
    H_corrected = -(probs_corrected * torch.log(probs_corrected + 1e-12)).sum(
        dim=-1, keepdim=True
    ) / math.log(vocab_size)

    gamma_time = math.exp(-gamma_decay * time_step)
    time_mult = (1 - gamma_time) / (gamma_time + 1e-12)

    # Entropy gate on lang prior strength
    gate_frac_txt = torch.sigmoid((H_corrected - hidden_eta) / hidden_tau)
    g_txt = time_mult * gate_frac_txt

    # Plausibility cutoff on corrected logits
    cutoff_th = math.log(beta_cutoff) + logits_corrected.max(dim=-1, keepdim=True).values
    logits_corrected = logits_corrected.masked_fill(logits_corrected < cutoff_th, -float("inf"))

    log_p_corrected = torch.log_softmax(logits_corrected, dim=-1)
    log_p_lang = torch.log_softmax(next_token_logits_lang, dim=-1)

    final = (1 + g_txt) * log_p_corrected - g_txt * log_p_lang

    return final, g_vis, g_txt


# ---------------------------------------------------------------------------
# Main sampler
# ---------------------------------------------------------------------------

def latent_sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfigLatent,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    r"""
    Latent-space sampling: HSC, LEG, or LLH decoding for hallucination mitigation.

    Dispatches to the appropriate core logic based on generation_config.latent_method.
    """
    latent_method = generation_config.latent_method  # "hsc", "leg", "llh"

    # ---- init values (same structure as entropygate_sample) ----
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    max_length = generation_config.max_length
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample

    # ---- init tuples ----
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

    # ---- Initialize variables ----
    pixel_values = model_kwargs.pop("pixel_values", None)
    key_position = generation_config.key_position

    # Lang Prior
    model_kwargs_lang_prior = copy.deepcopy(model_kwargs)
    input_ids_lang_prior = generation_config.input_ids_lang_prior
    model_kwargs_lang_prior["attention_mask"] = torch.ones_like(input_ids_lang_prior)

    # Stat Bias
    model_kwargs_stat_bias = copy.deepcopy(model_kwargs)

    # ---- Shared hyperparameters ----
    gamma_decay = generation_config.gamma_decay
    beta_cutoff_fixed = generation_config.beta_cutoff_fixed
    theta_safe = torch.tensor(generation_config.theta_safe, device=input_ids.device)

    # ---- Method-specific hyperparameters ----
    if latent_method == "hsc":
        alpha_base = generation_config.hsc_alpha_base
        alpha_min = generation_config.hsc_alpha_min
        eta = generation_config.hsc_eta
        tau = generation_config.hsc_tau
    elif latent_method == "leg":
        alpha_base = generation_config.alpha_base_vis
        alpha_min = generation_config.alpha_min_vis
        eta = generation_config.eta_vis
        tau = generation_config.tau_gate
        leg_hidden_layer = generation_config.leg_hidden_layer
    elif latent_method == "llh":
        hidden_alpha_base = generation_config.llh_hidden_alpha_base
        hidden_alpha_min = generation_config.llh_hidden_alpha_min
        hidden_eta = generation_config.llh_hidden_eta
        hidden_tau = generation_config.llh_hidden_tau

    # Which passes need hidden states
    need_hidden_orig = True   # all methods need hidden state from orig pass
    need_hidden_stat = latent_method in ("hsc", "llh")  # HSC/LLH need stat hidden state

    time_step = 1

    # ---- Logging accumulators ----
    _log_H_vals = []
    _log_g_vis_vals = []
    _log_g_txt_vals = []
    _log_skip_count = 0
    _log_gate_count = 0

    # ---- Sequence tracking ----
    batch_size, cur_len = input_ids.shape
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

    model_kwargs = _compat_get_initial_cache_position(self, input_ids, model_kwargs)
    model_kwargs_lang_prior = _compat_get_initial_cache_position(self, input_ids_lang_prior, model_kwargs_lang_prior)
    model_kwargs_stat_bias = _compat_get_initial_cache_position(self, input_ids, model_kwargs_stat_bias)

    # ==================================================================
    # Main decoding loop
    # ==================================================================

    while _compat_has_unfinished(self, this_peer_finished, synced_gpus, input_ids.device, cur_len, max_length):

        # ---- Forward pass 1: original model (with hidden states) ----
        outputs, model_kwargs = get_generations(self,
                                                input_ids,
                                                pixel_values=pixel_values,
                                                model_kwargs=model_kwargs,
                                                generation_config=generation_config,
                                                key_position=None,
                                                use_text_mask=False,
                                                use_fast_v=False,
                                                output_attentions=False,
                                                output_hidden_states=need_hidden_orig)

        if synced_gpus and this_peer_finished:
            continue

        next_token_logits = get_next_token_logits(outputs, input_ids)

        # Extract hidden state from orig pass
        h_orig = None
        if need_hidden_orig:
            if outputs.hidden_states is None:
                raise RuntimeError(
                    "output_hidden_states=True was requested but outputs.hidden_states is None. "
                    "Ensure the model forward supports output_hidden_states."
                )
            if latent_method == "leg":
                h_orig = get_hidden_state_at_layer(outputs, leg_hidden_layer)
            else:
                # HSC/LLH: use second-to-last element = pre-norm hidden state
                h_orig = get_hidden_state_at_layer(outputs, -2)

        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        generation_config.minimum_text_tokens = new_text_tokens(time_step)

        # ---- Forward pass 2: lang prior (text-only, no hidden states) ----
        outputs_lang_prior, model_kwargs_lang_prior = get_generations(self,
                                                input_ids_lang_prior,
                                                pixel_values=None,
                                                model_kwargs=model_kwargs_lang_prior,
                                                generation_config=generation_config,
                                                key_position=key_position,
                                                use_text_mask=True,
                                                use_fast_v=False,
                                                output_attentions=output_attentions,
                                                output_hidden_states=False)

        if synced_gpus and this_peer_finished:
            continue

        next_token_logits_lang_prior = get_next_token_logits(outputs_lang_prior, input_ids_lang_prior)
        del outputs_lang_prior
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---- Forward pass 3: stat bias (fast_v, with hidden states if needed) ----
        outputs_stat_bias, model_kwargs_stat_bias = get_generations(self,
                                                input_ids,
                                                pixel_values=pixel_values,
                                                model_kwargs=model_kwargs_stat_bias,
                                                generation_config=generation_config,
                                                key_position=key_position,
                                                use_text_mask=False,
                                                use_fast_v=True,
                                                output_attentions=output_attentions,
                                                output_hidden_states=need_hidden_stat)

        if synced_gpus and this_peer_finished:
            continue

        next_token_logits_stat_bias = get_next_token_logits(outputs_stat_bias, input_ids)

        # Extract hidden state from stat bias pass
        h_stat = None
        if need_hidden_stat:
            if outputs_stat_bias.hidden_states is None:
                raise RuntimeError(
                    "output_hidden_states=True was requested for stat_bias pass but "
                    "outputs.hidden_states is None."
                )
            h_stat = get_hidden_state_at_layer(outputs_stat_bias, -2)

        del outputs_stat_bias
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ==== Core logic ====

        # Entropy from original logits (used by HSC/LLH for gating,
        # and for safety skip check in all methods)
        probs_orig = torch.softmax(next_token_logits, dim=-1)
        vocab_size = next_token_logits.shape[-1]
        H_t = -(probs_orig * torch.log(probs_orig + 1e-12)).sum(dim=-1, keepdim=True) / math.log(vocab_size)
        max_prob = probs_orig.max(dim=-1, keepdim=True).values.item()

        # Safety skip: if model is very confident, don't intervene
        if max_prob > theta_safe.item():
            final_logits = next_token_logits
            _log_skip_count += 1
            _log_H_vals.append(H_t.item())
            _log_g_vis_vals.append(0.0)
            _log_g_txt_vals.append(0.0)
            logger.debug(
                f"step={time_step:03d} H_t={H_t.item():.4f} SKIP "
                f"(max_prob={max_prob:.4f}>{theta_safe.item():.2f}) method={latent_method}"
            )
        else:
            # Dispatch to method-specific core logic
            if latent_method == "hsc":
                final_logits, g_vis, g_txt = _hsc_core(
                    self, next_token_logits, next_token_logits_lang_prior,
                    h_orig, h_stat, H_t,
                    alpha_base, alpha_min, eta, tau,
                    gamma_decay, beta_cutoff_fixed, time_step)

            elif latent_method == "leg":
                final_logits, g_vis, g_txt = _leg_core(
                    self, next_token_logits, next_token_logits_lang_prior,
                    next_token_logits_stat_bias, h_orig,
                    alpha_base, alpha_min, eta, tau,
                    gamma_decay, beta_cutoff_fixed, time_step)

            elif latent_method == "llh":
                final_logits, g_vis, g_txt = _llh_core(
                    self, next_token_logits, next_token_logits_lang_prior,
                    h_orig, h_stat, H_t,
                    hidden_alpha_base, hidden_alpha_min, hidden_eta, hidden_tau,
                    gamma_decay, beta_cutoff_fixed, time_step)

            _log_gate_count += 1
            _log_H_vals.append(H_t.item())
            _log_g_vis_vals.append(g_vis.item() if torch.is_tensor(g_vis) else g_vis)
            _log_g_txt_vals.append(g_txt.item() if torch.is_tensor(g_txt) else g_txt)
            logger.debug(
                f"step={time_step:03d} H_t={H_t.item():.4f} "
                f"g_vis={g_vis.item() if torch.is_tensor(g_vis) else g_vis:.4f} "
                f"g_txt={g_txt.item() if torch.is_tensor(g_txt) else g_txt:.4f} "
                f"method={latent_method}"
            )

        time_step += 1

        # ---- Pre-process distribution ----
        next_token_scores = logits_processor(input_ids, final_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (final_logits,)

        # ---- Token selection ----
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # Finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # Update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        input_ids_lang_prior = torch.cat([input_ids_lang_prior, next_tokens[:, None]], dim=-1)

        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

    if streamer is not None:
        streamer.end()

    # ---- Per-sample summary log ----
    total_steps = _log_skip_count + _log_gate_count
    if total_steps > 0 and _log_H_vals:
        avg_H = sum(_log_H_vals) / len(_log_H_vals)
        avg_g_vis = sum(_log_g_vis_vals) / len(_log_g_vis_vals)
        avg_g_txt = sum(_log_g_txt_vals) / len(_log_g_txt_vals)
        skip_rate = _log_skip_count / total_steps
        logger.info(
            f"[{latent_method}_summary] steps={total_steps} "
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


def patch_latent_sampling():
    transformers.generation.utils.GenerationMixin._sample = latent_sample

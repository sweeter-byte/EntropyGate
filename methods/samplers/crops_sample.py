import math
import copy
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

from methods.generation_configs.contrastive_generation_config import GenerationConfigContrastive
from methods.utils.crops_samplers_utils import get_generations, get_next_token_logits
from methods.samplers.entropygate_sample import (
    _compat_get_initial_cache_position,
    _compat_has_unfinished,
)


def crops_sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfigContrastive,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    """CRoPS sampling ported to transformers 5.x using compat helpers."""
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

    #### Initialise variables for CRoPS ####
    pixel_values = model_kwargs.pop("pixel_values", None)
    key_position = generation_config.key_position

    # Lang Prior
    model_kwargs_lang_prior = copy.deepcopy(model_kwargs)
    input_ids_lang_prior = generation_config.input_ids_lang_prior
    lambda_lang_prior = generation_config.lambda_lang_prior
    model_kwargs_lang_prior["attention_mask"] = torch.ones_like(input_ids_lang_prior)

    # Stat Bias
    model_kwargs_stat_bias = copy.deepcopy(model_kwargs)
    alpha_stat_bias = generation_config.alpha_stat_bias

    # Other
    time_step = 1
    beta_cutoff = torch.tensor(generation_config.beta_cutoff)
    max_threshold_plausibility_constraint = torch.tensor(
        generation_config.max_threshold_plausibility_constraint
    )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

    model_kwargs = _compat_get_initial_cache_position(self, input_ids, model_kwargs)
    model_kwargs_lang_prior = _compat_get_initial_cache_position(self, input_ids_lang_prior, model_kwargs_lang_prior)
    model_kwargs_stat_bias = _compat_get_initial_cache_position(self, input_ids, model_kwargs_stat_bias)

    while _compat_has_unfinished(self, this_peer_finished, synced_gpus, input_ids.device, cur_len, max_length):

        outputs, model_kwargs = get_generations(self,
                                                input_ids,
                                                pixel_values=pixel_values,
                                                model_kwargs=model_kwargs,
                                                generation_config=generation_config,
                                                key_position=None,
                                                use_text_mask=False,
                                                use_fast_v=False,
                                                output_attentions=output_attentions,
                                                output_hidden_states=output_hidden_states)

        if synced_gpus and this_peer_finished:
            continue

        next_token_logits = get_next_token_logits(outputs, input_ids)

        generation_config.minimum_text_tokens = new_text_tokens(time_step)

        # logits with Language Prior
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

        # logits with Stat Bias
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

        # Apply cutoff threshold
        cutoff_th = torch.log(beta_cutoff) + next_token_logits.max(dim=-1, keepdim=True).values
        next_token_logits = next_token_logits.masked_fill(next_token_logits < cutoff_th, -float("inf"))

        log_probs_next_token = torch.log_softmax(next_token_logits, dim=-1)
        probs_next_token = torch.softmax(next_token_logits, dim=-1)

        if probs_next_token.max(dim=-1, keepdim=True).values > max_threshold_plausibility_constraint:
            final_logits = next_token_logits
        else:
            log_probs_next_token_lang_prior = torch.log_softmax(next_token_logits_lang_prior, dim=-1)
            log_probs_next_token_stat_bias = torch.log_softmax(next_token_logits_stat_bias, dim=-1)
            gamma_lang_prior = math.exp(-lambda_lang_prior * time_step)
            # Remove Language Prior
            final_logits = log_probs_next_token + \
                (1 - gamma_lang_prior) / gamma_lang_prior * (log_probs_next_token - log_probs_next_token_lang_prior)
            # Remove Stat Bias
            final_logits = (1 + alpha_stat_bias) * final_logits - alpha_stat_bias * log_probs_next_token_stat_bias

        time_step += 1

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, final_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (final_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)
            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

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

        del outputs, outputs_lang_prior, outputs_stat_bias

    if streamer is not None:
        streamer.end()

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


def patch_crops_sampling():
    transformers.generation.utils.GenerationMixin._sample = crops_sample


def new_text_tokens(t, b0=10, b1=30, lamb=0.001):
    return math.floor(b0 + b1 * (1 - math.exp(-lamb * t)))

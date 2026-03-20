"""Utility wrapper around get_generations that always returns hidden states.

For latent-space methods (HSC, LEG, LLH) we need the last hidden state from
certain forward passes.  This helper always requests output_hidden_states=True
and extracts the last-token hidden state from the result.
"""

import torch


def get_last_hidden_state(outputs, input_ids):
    """Extract the last-token hidden state from model outputs.

    For LlavaForConditionalGeneration, outputs.hidden_states is a tuple of
    (num_layers+1) tensors, each (batch, seq_len, hidden_dim).
    The last element is the final-layer hidden state AFTER norm.

    Returns: (batch, hidden_dim) tensor — the hidden state of the last token.
    """
    if outputs.hidden_states is None:
        raise RuntimeError(
            "outputs.hidden_states is None — did you forget output_hidden_states=True?"
        )
    # Last element = post-norm hidden state (same as what goes into lm_head)
    last_layer_hs = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
    return last_layer_hs[:, -1, :].clone().float()  # (batch, hidden_dim)


def get_hidden_state_at_layer(outputs, layer_idx):
    """Extract the last-token hidden state from a specific layer.

    layer_idx: int, can be negative (e.g. -2 for second-to-last layer).
    Returns: (batch, hidden_dim) tensor.
    """
    if outputs.hidden_states is None:
        raise RuntimeError(
            "outputs.hidden_states is None — did you forget output_hidden_states=True?"
        )
    layer_hs = outputs.hidden_states[layer_idx]  # (batch, seq_len, hidden_dim)
    return layer_hs[:, -1, :].clone().float()  # (batch, hidden_dim)


def hidden_state_to_logits(model, hidden_state):
    """Project a hidden state vector through the model's lm_head.

    model: the full model (e.g. LlavaForConditionalGeneration)
    hidden_state: (batch, hidden_dim) float tensor

    Returns: (batch, vocab_size) float tensor — logits.
    """
    # LlavaForConditionalGeneration wraps a language_model which has lm_head
    lm_head = None
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'lm_head'):
        lm_head = model.language_model.lm_head
    elif hasattr(model, 'lm_head'):
        lm_head = model.lm_head
    else:
        raise AttributeError("Cannot find lm_head on model")

    # lm_head expects (batch, seq_len, hidden_dim) — add seq_len=1 dim
    h = hidden_state.unsqueeze(1).to(dtype=next(lm_head.parameters()).dtype)
    logits = lm_head(h)  # (batch, 1, vocab_size)
    return logits[:, 0, :].float()  # (batch, vocab_size)

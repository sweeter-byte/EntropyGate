"""Utility functions for latent-space methods (HSC, LEG, LLH).

Provides helpers for:
- Extracting hidden states from model outputs
- Accessing the model's final norm layer and lm_head
- Projecting pre-norm hidden states to logits via norm + lm_head
"""

import torch


def get_last_hidden_state(outputs, input_ids):
    """Extract the last-token POST-NORM hidden state from model outputs.

    outputs.hidden_states[-1] is the post-norm hidden state (what lm_head
    directly operates on).  For a linear lm_head, contrast in this space
    is equivalent to logit-space contrast.

    Returns: (batch, hidden_dim) tensor.
    """
    if outputs.hidden_states is None:
        raise RuntimeError(
            "outputs.hidden_states is None — did you forget output_hidden_states=True?"
        )
    last_layer_hs = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
    return last_layer_hs[:, -1, :].clone().float()  # (batch, hidden_dim)


def get_hidden_state_at_layer(outputs, layer_idx):
    """Extract the last-token hidden state from a specific layer.

    layer_idx indexes into outputs.hidden_states:
      -1  = post-norm (same as get_last_hidden_state)
      -2  = last transformer layer input (second-to-last layer output)
      -16 = roughly mid-layer for 32-layer models
      etc.

    Returns: (batch, hidden_dim) tensor.
    """
    if outputs.hidden_states is None:
        raise RuntimeError(
            "outputs.hidden_states is None — did you forget output_hidden_states=True?"
        )
    layer_hs = outputs.hidden_states[layer_idx]  # (batch, seq_len, hidden_dim)
    return layer_hs[:, -1, :].clone().float()  # (batch, hidden_dim)


def hidden_state_to_logits(model, hidden_state):
    """Project a POST-NORM hidden state through lm_head only.

    Use this when you already have a post-norm hidden state.
    For pre-norm hidden states, use project_prenorm_to_logits instead.

    model: the full model (e.g. LlavaForConditionalGeneration)
    hidden_state: (batch, hidden_dim) float tensor

    Returns: (batch, vocab_size) float tensor — logits.
    """
    _, lm_head = get_norm_and_lm_head(model)
    h = hidden_state.unsqueeze(1).to(dtype=next(lm_head.parameters()).dtype)
    logits = lm_head(h)  # (batch, 1, vocab_size)
    return logits[:, 0, :].float()


def get_norm_and_lm_head(model):
    """
    Ultra-robust version: automatically find norm and lm_head.
    """

    # -------- lm_head --------
    if hasattr(model, "lm_head"):
        lm_head = model.lm_head
    elif hasattr(model, "get_output_embeddings"):
        lm_head = model.get_output_embeddings()
    else:
        raise AttributeError("Cannot find lm_head on model.")

    # -------- 常见路径优先 --------
    candidates = [
        "model.norm",
        "model.final_layernorm",
        "transformer.norm",
        "transformer.final_layernorm",
        "norm",
    ]

    for path in candidates:
        obj = model
        found = True
        for attr in path.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                found = False
                break
        if found:
            return obj, lm_head

    # -------- fallback：自动搜索 --------
    for name, module in model.named_modules():
        if "norm" in name.lower():
            # 只取最后一个（通常是final norm）
            last_norm = module

    if 'last_norm' in locals():
        print(f"[DEBUG] Using detected norm: {name}")
        return last_norm, lm_head

    raise AttributeError("Cannot find norm in model.")
    
def project_prenorm_to_logits(model, hidden_state):
    """Apply norm + lm_head to a PRE-NORM hidden state to get logits.

    This is the key operation for latent-space contrast: the hidden state
    from outputs.hidden_states[-2] is pre-norm (not yet processed by
    the model's final RMSNorm).  After contrast in this space, we apply
    norm + lm_head.  Because RMSNorm is non-linear (divides by L2-norm),
    this gives DIFFERENT results from logit-space contrast.

    model: the full model
    hidden_state: (batch, hidden_dim) float tensor (pre-norm)

    Returns: (batch, vocab_size) float tensor — logits.
    """
    norm, lm_head = get_norm_and_lm_head(model)
    param_dtype = next(lm_head.parameters()).dtype

    h = hidden_state.unsqueeze(1).to(dtype=param_dtype)  # (batch, 1, hidden_dim)
    h = norm(h)        # RMSNorm — non-linear!
    logits = lm_head(h)  # (batch, 1, vocab_size)
    return logits[:, 0, :].float()

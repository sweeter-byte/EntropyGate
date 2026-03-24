# =============================================================================
# EDGE: Entropy-Driven Gated Decoding — Hyperparameter Constants
#
# Nested contrastive formula (E5):
#   intermediate = log_p + (1-gamma_t)/gamma_t * (log_p - log_p_lang)
#   final = (1 + g_vis) * intermediate - g_vis * log_p_stat
#   g_vis = alpha_min_vis + (alpha_base_vis - alpha_min_vis) * sigmoid((H_t - eta_vis) / tau)
# =============================================================================

# ---- CRoPS-inherited structural parameters ----

KEY_POSITION = "key_position"

AGGREGATE_LAYER_FAST_V = "aggregate_layer_fast_v"
DEFAULT_AGGREGATE_LAYER_FAST_V = 2

MINUMUM_FAST_V_TOKENS = "minumum_fast_v_tokens"
DEFAULT_MINUMUM_FAST_V_TOKENS = 50

AGGREGATE_LAYER_TEXT_MASK = "aggregate_layer_text_mask"
DEFAULT_AGGREGATE_LAYER_TEXT_MASK = 2

MINIMUM_TEXT_TOKENS = "minimum_text_tokens"
DEFAULT_MINIMUM_TEXT_TOKENS = 50

INPUT_IDS_LANG_PRIOR = "input_ids_lang_prior"

# ---- EDGE core parameters (E5 best config) ----

# Visual entropy gate: g_vis range [alpha_min_vis, alpha_base_vis]
ALPHA_BASE_VIS = "alpha_base_vis"
DEFAULT_ALPHA_BASE_VIS = 1.5

ALPHA_MIN_VIS = "alpha_min_vis"
DEFAULT_ALPHA_MIN_VIS = 0.5

# Entropy threshold for visual gating
ETA_VIS = "eta_vis"
DEFAULT_ETA_VIS = 0.1

# Gate temperature (controls sigmoid sharpness)
TAU_GATE = "tau_gate"
DEFAULT_TAU_GATE = 0.05

# Time decay coefficient for language prior (CRoPS lambda)
GAMMA_DECAY = "gamma_decay"
DEFAULT_GAMMA_DECAY = 0.01

# Fixed cutoff threshold (log-domain plausibility filter)
BETA_CUTOFF = "beta_cutoff"
DEFAULT_BETA_CUTOFF = 0.1

# Safety skip threshold: skip contrastive when max_prob > theta_safe
THETA_SAFE = "theta_safe"
DEFAULT_THETA_SAFE = 0.99

# ---- Generation defaults ----

DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 1
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 40

# ---- Image token IDs per model ----

_BACKBONE_IMAGE_TOKEN_IDS = {
    "llava-1.5-7b-hf": 32000,
    "llama3-llava-next-8b-hf": 128256,
    "llava-v1.6-vicuna-7b-hf": 32000,
    "llava-v1.6-vicuna-13b-hf": 128256,
    "llava-1.5-13b-hf": 32000,
    "Qwen2.5-VL-7B-Instruct": 151655,
    "R1-Onevision-7B": 151655,
    "Vision-R1-7B": 151655,
    "VL-Rethinker-7B": 151655,
    "VL-Cogito": 151655,
    "OpenVLThinker-7B": 151655,
}

# Legacy exact-match dict
BACKBONE_IMAGE_TOKEN_IDS = {
    "llava-hf/llava-1.5-7b-hf": 32000,
    "llava-hf/llama3-llava-next-8b-hf": 128256,
    "llava-hf/llava-v1.6-vicuna-7b-hf": 32000,
    "llava-hf/llava-v1.6-vicuna-13b-hf": 128256,
}


def get_image_token_id(model_name: str) -> int:
    """Resolve image token ID from model name or local path."""
    if model_name in BACKBONE_IMAGE_TOKEN_IDS:
        return BACKBONE_IMAGE_TOKEN_IDS[model_name]

    short_name = model_name.rstrip("/").split("/")[-1]
    if short_name in _BACKBONE_IMAGE_TOKEN_IDS:
        return _BACKBONE_IMAGE_TOKEN_IDS[short_name]

    for key, token_id in _BACKBONE_IMAGE_TOKEN_IDS.items():
        if key in model_name:
            return token_id

    raise KeyError(
        f"Cannot find image token ID for model '{model_name}'. "
        f"Known models: {list(_BACKBONE_IMAGE_TOKEN_IDS.keys())}"
    )

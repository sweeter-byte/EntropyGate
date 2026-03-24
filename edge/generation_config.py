"""EDGE generation config — extends HuggingFace GenerationConfig with EDGE parameters."""

from transformers import GenerationConfig

from edge.constants import (
    KEY_POSITION,
    AGGREGATE_LAYER_FAST_V, DEFAULT_AGGREGATE_LAYER_FAST_V,
    MINUMUM_FAST_V_TOKENS, DEFAULT_MINUMUM_FAST_V_TOKENS,
    AGGREGATE_LAYER_TEXT_MASK, DEFAULT_AGGREGATE_LAYER_TEXT_MASK,
    MINIMUM_TEXT_TOKENS, DEFAULT_MINIMUM_TEXT_TOKENS,
    INPUT_IDS_LANG_PRIOR,
    ALPHA_BASE_VIS, DEFAULT_ALPHA_BASE_VIS,
    ALPHA_MIN_VIS, DEFAULT_ALPHA_MIN_VIS,
    ETA_VIS, DEFAULT_ETA_VIS,
    TAU_GATE, DEFAULT_TAU_GATE,
    GAMMA_DECAY, DEFAULT_GAMMA_DECAY,
    BETA_CUTOFF, DEFAULT_BETA_CUTOFF,
    THETA_SAFE, DEFAULT_THETA_SAFE,
)


class EdgeGenerationConfig(GenerationConfig):

    def __init__(self, **kwargs):
        # Pop EDGE-specific keys BEFORE super().__init__() consumes kwargs
        self.key_position = kwargs.pop(KEY_POSITION, None)

        # Fast V (image token pruning)
        self.aggregate_layer_fast_v = kwargs.pop(AGGREGATE_LAYER_FAST_V, DEFAULT_AGGREGATE_LAYER_FAST_V)
        self.minumum_fast_v_tokens = kwargs.pop(MINUMUM_FAST_V_TOKENS, DEFAULT_MINUMUM_FAST_V_TOKENS)

        # Text Mask (text token selection)
        self.aggregate_layer_text_mask = kwargs.pop(AGGREGATE_LAYER_TEXT_MASK, DEFAULT_AGGREGATE_LAYER_TEXT_MASK)
        self.minimum_text_tokens = kwargs.pop(MINIMUM_TEXT_TOKENS, DEFAULT_MINIMUM_TEXT_TOKENS)

        # Lang Prior input_ids
        self.input_ids_lang_prior = kwargs.pop(INPUT_IDS_LANG_PRIOR, None)

        # ---- EDGE core parameters ----
        self.alpha_base_vis = kwargs.pop(ALPHA_BASE_VIS, DEFAULT_ALPHA_BASE_VIS)
        self.alpha_min_vis = kwargs.pop(ALPHA_MIN_VIS, DEFAULT_ALPHA_MIN_VIS)
        self.eta_vis = kwargs.pop(ETA_VIS, DEFAULT_ETA_VIS)
        self.tau_gate = kwargs.pop(TAU_GATE, DEFAULT_TAU_GATE)
        self.gamma_decay = kwargs.pop(GAMMA_DECAY, DEFAULT_GAMMA_DECAY)
        self.beta_cutoff = kwargs.pop(BETA_CUTOFF, DEFAULT_BETA_CUTOFF)
        self.theta_safe = kwargs.pop(THETA_SAFE, DEFAULT_THETA_SAFE)

        super().__init__(**kwargs)

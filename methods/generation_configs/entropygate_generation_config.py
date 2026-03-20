from transformers import GenerationConfig

from constants.crops_constants import (
    KEY_POSITION,
    AGGREGATE_LAYER_FAST_V,
    MINUMUM_FAST_V_TOKENS,
    AGGREGATE_LAYER_TEXT_MASK,
    MINIMUM_TEXT_TOKENS,
    INPUT_IDS_LANG_PRIOR,
    DEFAULT_AGGREGATE_LAYER_FAST_V,
    DEFAULT_MINUMUM_FAST_V_TOKENS,
    DEFAULT_AGGREGATE_LAYER_TEXT_MASK,
    DEFAULT_MINIMUM_TEXT_TOKENS,
)

from constants.entropygate_constants import (
    ALPHA_BASE_VIS, DEFAULT_ALPHA_BASE_VIS,
    ALPHA_BASE_TXT, DEFAULT_ALPHA_BASE_TXT,
    ALPHA_MIN_VIS, DEFAULT_ALPHA_MIN_VIS,
    ALPHA_MIN_TXT, DEFAULT_ALPHA_MIN_TXT,
    ETA_VIS, DEFAULT_ETA_VIS,
    ETA_TXT, DEFAULT_ETA_TXT,
    TAU_GATE, DEFAULT_TAU_GATE,
    GAMMA_DECAY, DEFAULT_GAMMA_DECAY,
    BETA_BASE, DEFAULT_BETA_BASE,
    BETA_RANGE, DEFAULT_BETA_RANGE,
    THETA_SAFE, DEFAULT_THETA_SAFE,
    EG_SCHEME, DEFAULT_EG_SCHEME,
    BETA_CUTOFF_FIXED, DEFAULT_BETA_CUTOFF_FIXED,
    THETA_SAFE_ALIGNED, DEFAULT_THETA_SAFE_ALIGNED,
    TIME_DECAY_MODE, DEFAULT_TIME_DECAY_MODE,
    ALPHA_TIME_TXT, DEFAULT_ALPHA_TIME_TXT,
    ADAPTIVE_ETA, DEFAULT_ADAPTIVE_ETA,
    ETA_EMA_MOMENTUM, DEFAULT_ETA_EMA_MOMENTUM,
    ETA_VIS_OFFSET, DEFAULT_ETA_VIS_OFFSET,
    ETA_TXT_OFFSET, DEFAULT_ETA_TXT_OFFSET,
    SOFT_SUPPRESS, DEFAULT_SOFT_SUPPRESS,
    SOFT_SUPPRESS_K, DEFAULT_SOFT_SUPPRESS_K,
)


class GenerationConfigEntropyGate(GenerationConfig):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.key_position = kwargs.pop(KEY_POSITION, None)

        # Fast V (reused from CRoPS)
        self.aggregate_layer_fast_v = kwargs.pop(AGGREGATE_LAYER_FAST_V, DEFAULT_AGGREGATE_LAYER_FAST_V)
        self.minumum_fast_v_tokens = kwargs.pop(MINUMUM_FAST_V_TOKENS, DEFAULT_MINUMUM_FAST_V_TOKENS)

        # Text Mask (reused from CRoPS)
        self.aggregate_layer_text_mask = kwargs.pop(AGGREGATE_LAYER_TEXT_MASK, DEFAULT_AGGREGATE_LAYER_TEXT_MASK)
        self.minimum_text_tokens = kwargs.pop(MINIMUM_TEXT_TOKENS, DEFAULT_MINIMUM_TEXT_TOKENS)

        # Lang Prior input_ids (reused from CRoPS)
        self.input_ids_lang_prior = kwargs.pop(INPUT_IDS_LANG_PRIOR, None)

        # ---- EntropyGate-specific parameters ----
        self.alpha_base_vis = kwargs.pop(ALPHA_BASE_VIS, DEFAULT_ALPHA_BASE_VIS)
        self.alpha_base_txt = kwargs.pop(ALPHA_BASE_TXT, DEFAULT_ALPHA_BASE_TXT)
        self.alpha_min_vis = kwargs.pop(ALPHA_MIN_VIS, DEFAULT_ALPHA_MIN_VIS)
        self.alpha_min_txt = kwargs.pop(ALPHA_MIN_TXT, DEFAULT_ALPHA_MIN_TXT)
        self.eta_vis = kwargs.pop(ETA_VIS, DEFAULT_ETA_VIS)
        self.eta_txt = kwargs.pop(ETA_TXT, DEFAULT_ETA_TXT)
        self.tau_gate = kwargs.pop(TAU_GATE, DEFAULT_TAU_GATE)
        self.gamma_decay = kwargs.pop(GAMMA_DECAY, DEFAULT_GAMMA_DECAY)
        self.beta_base = kwargs.pop(BETA_BASE, DEFAULT_BETA_BASE)
        self.beta_range = kwargs.pop(BETA_RANGE, DEFAULT_BETA_RANGE)
        self.theta_safe = kwargs.pop(THETA_SAFE, DEFAULT_THETA_SAFE)
        self.eg_scheme = kwargs.pop(EG_SCHEME, DEFAULT_EG_SCHEME)
        self.beta_cutoff_fixed = kwargs.pop(BETA_CUTOFF_FIXED, DEFAULT_BETA_CUTOFF_FIXED)
        self.theta_safe_aligned = kwargs.pop(THETA_SAFE_ALIGNED, DEFAULT_THETA_SAFE_ALIGNED)
        self.time_decay_mode = kwargs.pop(TIME_DECAY_MODE, DEFAULT_TIME_DECAY_MODE)
        self.alpha_time_txt = kwargs.pop(ALPHA_TIME_TXT, DEFAULT_ALPHA_TIME_TXT)
        self.adaptive_eta = kwargs.pop(ADAPTIVE_ETA, DEFAULT_ADAPTIVE_ETA)
        self.eta_ema_momentum = kwargs.pop(ETA_EMA_MOMENTUM, DEFAULT_ETA_EMA_MOMENTUM)
        self.eta_vis_offset = kwargs.pop(ETA_VIS_OFFSET, DEFAULT_ETA_VIS_OFFSET)
        self.eta_txt_offset = kwargs.pop(ETA_TXT_OFFSET, DEFAULT_ETA_TXT_OFFSET)
        self.soft_suppress = kwargs.pop(SOFT_SUPPRESS, DEFAULT_SOFT_SUPPRESS)
        self.soft_suppress_k = kwargs.pop(SOFT_SUPPRESS_K, DEFAULT_SOFT_SUPPRESS_K)

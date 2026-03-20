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
    ALPHA_MIN_VIS, DEFAULT_ALPHA_MIN_VIS,
    ETA_VIS, DEFAULT_ETA_VIS,
    TAU_GATE, DEFAULT_TAU_GATE,
    GAMMA_DECAY, DEFAULT_GAMMA_DECAY,
    BETA_CUTOFF_FIXED, DEFAULT_BETA_CUTOFF_FIXED,
    THETA_SAFE, DEFAULT_THETA_SAFE,
)

from constants.latent_constants import (
    HSC_ALPHA_BASE, DEFAULT_HSC_ALPHA_BASE,
    HSC_ALPHA_MIN, DEFAULT_HSC_ALPHA_MIN,
    HSC_ETA, DEFAULT_HSC_ETA,
    HSC_TAU, DEFAULT_HSC_TAU,
    LEG_HIDDEN_LAYER, DEFAULT_LEG_HIDDEN_LAYER,
    LLH_HIDDEN_ALPHA_BASE, DEFAULT_LLH_HIDDEN_ALPHA_BASE,
    LLH_HIDDEN_ALPHA_MIN, DEFAULT_LLH_HIDDEN_ALPHA_MIN,
    LLH_HIDDEN_ETA, DEFAULT_LLH_HIDDEN_ETA,
    LLH_HIDDEN_TAU, DEFAULT_LLH_HIDDEN_TAU,
    LATENT_METHOD, DEFAULT_LATENT_METHOD,
)


class GenerationConfigLatent(GenerationConfig):
    """Generation config for latent-space methods (HSC, LEG, LLH).

    Inherits shared CRoPS params (key_position, fast_v, text_mask, input_ids_lang_prior)
    and adds latent-space specific params.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ---- Shared CRoPS params (reused from EntropyGate) ----
        self.key_position = kwargs.pop(KEY_POSITION, None)
        self.aggregate_layer_fast_v = kwargs.pop(AGGREGATE_LAYER_FAST_V, DEFAULT_AGGREGATE_LAYER_FAST_V)
        self.minumum_fast_v_tokens = kwargs.pop(MINUMUM_FAST_V_TOKENS, DEFAULT_MINUMUM_FAST_V_TOKENS)
        self.aggregate_layer_text_mask = kwargs.pop(AGGREGATE_LAYER_TEXT_MASK, DEFAULT_AGGREGATE_LAYER_TEXT_MASK)
        self.minimum_text_tokens = kwargs.pop(MINIMUM_TEXT_TOKENS, DEFAULT_MINIMUM_TEXT_TOKENS)
        self.input_ids_lang_prior = kwargs.pop(INPUT_IDS_LANG_PRIOR, None)

        # ---- EntropyGate params reused by LEG/LLH ----
        self.alpha_base_vis = kwargs.pop(ALPHA_BASE_VIS, DEFAULT_ALPHA_BASE_VIS)
        self.alpha_min_vis = kwargs.pop(ALPHA_MIN_VIS, DEFAULT_ALPHA_MIN_VIS)
        self.eta_vis = kwargs.pop(ETA_VIS, DEFAULT_ETA_VIS)
        self.tau_gate = kwargs.pop(TAU_GATE, DEFAULT_TAU_GATE)
        self.gamma_decay = kwargs.pop(GAMMA_DECAY, DEFAULT_GAMMA_DECAY)
        self.beta_cutoff_fixed = kwargs.pop(BETA_CUTOFF_FIXED, DEFAULT_BETA_CUTOFF_FIXED)
        self.theta_safe = kwargs.pop(THETA_SAFE, DEFAULT_THETA_SAFE)

        # ---- HSC-specific ----
        self.hsc_alpha_base = kwargs.pop(HSC_ALPHA_BASE, DEFAULT_HSC_ALPHA_BASE)
        self.hsc_alpha_min = kwargs.pop(HSC_ALPHA_MIN, DEFAULT_HSC_ALPHA_MIN)
        self.hsc_eta = kwargs.pop(HSC_ETA, DEFAULT_HSC_ETA)
        self.hsc_tau = kwargs.pop(HSC_TAU, DEFAULT_HSC_TAU)

        # ---- LEG-specific ----
        self.leg_hidden_layer = kwargs.pop(LEG_HIDDEN_LAYER, DEFAULT_LEG_HIDDEN_LAYER)

        # ---- LLH-specific ----
        self.llh_hidden_alpha_base = kwargs.pop(LLH_HIDDEN_ALPHA_BASE, DEFAULT_LLH_HIDDEN_ALPHA_BASE)
        self.llh_hidden_alpha_min = kwargs.pop(LLH_HIDDEN_ALPHA_MIN, DEFAULT_LLH_HIDDEN_ALPHA_MIN)
        self.llh_hidden_eta = kwargs.pop(LLH_HIDDEN_ETA, DEFAULT_LLH_HIDDEN_ETA)
        self.llh_hidden_tau = kwargs.pop(LLH_HIDDEN_TAU, DEFAULT_LLH_HIDDEN_TAU)

        # ---- Method selector ----
        self.latent_method = kwargs.pop(LATENT_METHOD, DEFAULT_LATENT_METHOD)

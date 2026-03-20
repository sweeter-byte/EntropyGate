# ---- Latent-space method hyperparameters ----
# Shared by HSC, LEG, LLH methods
#
# NOTE on entropy threshold (η):
# LASER (arxiv 2601.06803) uses η=0.8 (normalized entropy) for intervention,
# only acting when the model is truly confused.  Our default η=0.10 is more
# aggressive (gate saturates at H_t ≈ 0.25).  Sweeps should explore both
# regimes: low η (aggressive, always-on) and high η (LASER-inspired,
# intervene-only-when-confused).

# HSC (Hidden State Contrastive): contrast in hidden state space
# then project back via norm + lm_head.
# Uses outputs.hidden_states[-2] (second-to-last in tuple = last transformer
# layer's INPUT, i.e. second-to-last layer's OUTPUT).  The RMSNorm after
# contrast introduces non-linearity, making this different from logit-space
# contrast.
HSC_ALPHA_BASE = "hsc_alpha_base"
DEFAULT_HSC_ALPHA_BASE = 1.0

HSC_ALPHA_MIN = "hsc_alpha_min"
DEFAULT_HSC_ALPHA_MIN = 0.3

HSC_ETA = "hsc_eta"
DEFAULT_HSC_ETA = 0.10

HSC_TAU = "hsc_tau"
DEFAULT_HSC_TAU = 0.05

# LEG (Latent Entropy Gate): compute entropy from a MIDDLE layer's hidden
# state (projected through norm + lm_head) instead of from output logits.
# The intuition is that mid-layer uncertainty might predict hallucination
# risk earlier / differently than output-layer entropy.
#
# leg_hidden_layer is an index into outputs.hidden_states:
#   For a 32-layer model, outputs.hidden_states has 33 elements:
#     [0] = embeddings, [1]..[31] = layer 0..30 output, [32] = post-norm
#   -1  = post-norm (SAME entropy as logit, useless for LEG)
#   -2  = layer 30 output (very close to output, small difference)
#   -16 = layer 17 output (roughly middle of 32-layer model)
LEG_HIDDEN_LAYER = "leg_hidden_layer"
DEFAULT_LEG_HIDDEN_LAYER = -16  # mid-layer for 32-layer models

# LEG reuses alpha_base_vis, alpha_min_vis, eta_vis, tau_gate from entropygate_constants
# The only new thing is where entropy is computed (hidden state vs logit)

# LLH (Latent-Logit Hybrid): stat bias contrast in hidden space, lang prior
# in logit space with entropy-gated strength (not just time decay).
# Two-stage: hidden-space stat correction + entropy-modulated lang prior.
LLH_HIDDEN_ALPHA_BASE = "llh_hidden_alpha_base"
DEFAULT_LLH_HIDDEN_ALPHA_BASE = 1.0

LLH_HIDDEN_ALPHA_MIN = "llh_hidden_alpha_min"
DEFAULT_LLH_HIDDEN_ALPHA_MIN = 0.3

LLH_HIDDEN_ETA = "llh_hidden_eta"
DEFAULT_LLH_HIDDEN_ETA = 0.10

LLH_HIDDEN_TAU = "llh_hidden_tau"
DEFAULT_LLH_HIDDEN_TAU = 0.05

# LLH reuses gamma_decay, beta_cutoff_fixed, theta_safe from entropygate_constants

# Latent method selector
LATENT_METHOD = "latent_method"
DEFAULT_LATENT_METHOD = "hsc"  # "hsc", "leg", "llh"

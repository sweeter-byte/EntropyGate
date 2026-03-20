# ---- Latent-space method hyperparameters ----
# Shared by HSC, LEG, LLH methods

# HSC (Hidden State Contrastive): contrast strength in hidden state space
HSC_ALPHA_BASE = "hsc_alpha_base"
DEFAULT_HSC_ALPHA_BASE = 1.0

HSC_ALPHA_MIN = "hsc_alpha_min"
DEFAULT_HSC_ALPHA_MIN = 0.3

HSC_ETA = "hsc_eta"
DEFAULT_HSC_ETA = 0.10

HSC_TAU = "hsc_tau"
DEFAULT_HSC_TAU = 0.05

# LEG (Latent Entropy Gate): use hidden state entropy instead of logit entropy
LEG_HIDDEN_LAYER = "leg_hidden_layer"
DEFAULT_LEG_HIDDEN_LAYER = -1  # -1 means last layer before norm

# LEG reuses alpha_base_vis, alpha_min_vis, eta_vis, tau_gate from entropygate_constants
# The only new thing is where entropy is computed (hidden state vs logit)

# LLH (Latent-Logit Hybrid): stat bias contrast in hidden space, lang prior in logit space
LLH_HIDDEN_ALPHA_BASE = "llh_hidden_alpha_base"
DEFAULT_LLH_HIDDEN_ALPHA_BASE = 1.0

LLH_HIDDEN_ALPHA_MIN = "llh_hidden_alpha_min"
DEFAULT_LLH_HIDDEN_ALPHA_MIN = 0.3

LLH_HIDDEN_ETA = "llh_hidden_eta"
DEFAULT_LLH_HIDDEN_ETA = 0.10

LLH_HIDDEN_TAU = "llh_hidden_tau"
DEFAULT_LLH_HIDDEN_TAU = 0.05

# LLH reuses gamma_decay, beta_cutoff_fixed from entropygate_constants for lang prior part

# Latent method selector
LATENT_METHOD = "latent_method"
DEFAULT_LATENT_METHOD = "hsc"  # "hsc", "leg", "llh"

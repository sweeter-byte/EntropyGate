# ---- EntropyGate-specific hyperparameters ----

# Entropy gate: visual contrast base strength (corresponds to CRoPS α^(1))
ALPHA_BASE_VIS = "alpha_base_vis"
DEFAULT_ALPHA_BASE_VIS = 1.0

# Entropy gate: text contrast base strength (corresponds to CRoPS α_t^(2) base)
ALPHA_BASE_TXT = "alpha_base_txt"
DEFAULT_ALPHA_BASE_TXT = 1.0

# Entropy gate: visual contrast minimum floor (scheme D)
ALPHA_MIN_VIS = "alpha_min_vis"
DEFAULT_ALPHA_MIN_VIS = 0.0

# Entropy gate: text contrast minimum floor (scheme D)
ALPHA_MIN_TXT = "alpha_min_txt"
DEFAULT_ALPHA_MIN_TXT = 0.0

# Entropy threshold for visual gating
ETA_VIS = "eta_vis"
DEFAULT_ETA_VIS = 0.3

# Entropy threshold for text gating
ETA_TXT = "eta_txt"
DEFAULT_ETA_TXT = 0.4

# Gate temperature (controls sigmoid sharpness)
TAU_GATE = "tau_gate"
DEFAULT_TAU_GATE = 0.05

# Time decay coefficient (reuses CRoPS lambda_lang_prior concept)
GAMMA_DECAY = "gamma_decay"
DEFAULT_GAMMA_DECAY = 0.01

# Adaptive cutoff base value
BETA_BASE = "beta_base"
DEFAULT_BETA_BASE = 0.05

# Adaptive cutoff range
BETA_RANGE = "beta_range"
DEFAULT_BETA_RANGE = 0.15

# Safety skip threshold (softened plausibility constraint)
THETA_SAFE = "theta_safe"
DEFAULT_THETA_SAFE = 0.99

# Time decay mode for g_txt (direction 2)
# "multiply" = original: g_txt *= time_decay (front-loaded suppression)
# "additive" = g_txt = base_gate + alpha_time_txt * time_decay (always-on + bonus)
TIME_DECAY_MODE = "time_decay_mode"
DEFAULT_TIME_DECAY_MODE = "multiply"

# Additive time bonus strength for g_txt (only used when time_decay_mode="additive")
ALPHA_TIME_TXT = "alpha_time_txt"
DEFAULT_ALPHA_TIME_TXT = 1.0

# Adaptive eta mode (direction 3)
# False = fixed eta (default)
# True  = eta tracks running EMA of H_t
ADAPTIVE_ETA = "adaptive_eta"
DEFAULT_ADAPTIVE_ETA = False

# EMA momentum for adaptive eta
ETA_EMA_MOMENTUM = "eta_ema_momentum"
DEFAULT_ETA_EMA_MOMENTUM = 0.1

# Adaptive eta offset: eta = H_mean + offset * H_std
ETA_VIS_OFFSET = "eta_vis_offset"
DEFAULT_ETA_VIS_OFFSET = 0.5

ETA_TXT_OFFSET = "eta_txt_offset"
DEFAULT_ETA_TXT_OFFSET = 1.0

# Soft suppress mode (direction 4)
# False = hard theta_safe skip (default)
# True  = smooth confidence-based suppression
SOFT_SUPPRESS = "soft_suppress"
DEFAULT_SOFT_SUPPRESS = False

# Exponent for soft suppress: suppress_factor = (1 - max_prob^k)
SOFT_SUPPRESS_K = "soft_suppress_k"
DEFAULT_SOFT_SUPPRESS_K = 4.0

# Contrastive scheme selector
# "flat"    = original EntropyGate flat formula (Scheme D/D+A)
# "nested"  = Scheme E: CRoPS nested structure + entropy gate on stat bias
# "acd"     = Scheme F: ACD-style entropy ratio gating + nested structure
# "nested_aligned" = Scheme G: Scheme E + aligned cutoff/safety params
EG_SCHEME = "eg_scheme"
DEFAULT_EG_SCHEME = "flat"

# Fixed cutoff for Scheme G (aligned with CRoPS)
BETA_CUTOFF_FIXED = "beta_cutoff_fixed"
DEFAULT_BETA_CUTOFF_FIXED = 0.1

# Fixed safety threshold for Scheme G (aligned with CRoPS)
THETA_SAFE_ALIGNED = "theta_safe_aligned"
DEFAULT_THETA_SAFE_ALIGNED = 0.95

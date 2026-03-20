# ---- VCD-specific hyperparameters ----

# Contrastive decoding alpha (controls contrast strength)
VCD_ALPHA = "vcd_alpha"
DEFAULT_VCD_ALPHA = 1.0

# Adaptive plausibility constraint beta
VCD_BETA = "vcd_beta"
DEFAULT_VCD_BETA = 0.1

# Diffusion noise step (0-999, higher = more noise)
VCD_NOISE_STEP = "vcd_noise_step"
DEFAULT_VCD_NOISE_STEP = 500

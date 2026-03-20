from transformers import GenerationConfig

from constants.vcd_constants import (
    VCD_ALPHA, DEFAULT_VCD_ALPHA,
    VCD_BETA, DEFAULT_VCD_BETA,
    VCD_NOISE_STEP, DEFAULT_VCD_NOISE_STEP,
)


class GenerationConfigVCD(GenerationConfig):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.vcd_alpha = kwargs.pop(VCD_ALPHA, DEFAULT_VCD_ALPHA)
        self.vcd_beta = kwargs.pop(VCD_BETA, DEFAULT_VCD_BETA)
        self.vcd_noise_step = kwargs.pop(VCD_NOISE_STEP, DEFAULT_VCD_NOISE_STEP)

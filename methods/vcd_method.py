from methods.samplers.vcd_sample import patch_vcd_sampling


def patch_everything_vcd():
    """Patch sampling for VCD. No model forward patch needed —
    VCD uses the standard model forward, only the sampling loop changes."""
    patch_vcd_sampling()

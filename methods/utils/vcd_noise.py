import torch


def add_diffusion_noise(image_tensor, noise_step):
    """Add diffusion-style Gaussian noise to image tensor.

    Ported from VCD (DAMO-NLP-SG/VCD) vcd_utils/vcd_add_noise.py.
    Uses a sigmoid beta schedule with 1000 total steps.

    Args:
        image_tensor: Image tensor of shape (B, C, H, W) or (C, H, W).
        noise_step: Integer in [0, 999]. Higher = more noise.

    Returns:
        Noisy image tensor with the same shape and device.
    """
    num_steps = 1000

    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    noise = torch.randn_like(image_tensor)
    noisy_image = (
        alphas_bar_sqrt[noise_step] * image_tensor
        + one_minus_alphas_bar_sqrt[noise_step] * noise
    )
    return noisy_image

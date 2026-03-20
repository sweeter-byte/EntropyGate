_BACKBONE_IMAGE_TOKEN_IDS = {
    "llava-1.5-7b-hf": 32000,
    "llama3-llava-next-8b-hf": 128256,
    "llava-v1.6-vicuna-7b-hf": 32000,
    "llava-v1.6-vicuna-13b-hf": 128256,
    "llava-1.5-13b-hf": 32000,
}

# Legacy exact-match dict (kept for backward compatibility)
BACKBONE_IMAGE_TOKEN_IDS = {
    "llava-hf/llava-1.5-7b-hf": 32000,
    "llava-hf/llama3-llava-next-8b-hf": 128256,
    "llava-hf/llava-v1.6-vicuna-7b-hf": 32000,
    "llava-hf/llava-v1.6-vicuna-13b-hf": 128256,
}


def get_image_token_id(model_name: str) -> int:
    """Resolve image token ID from model name or local path.

    Tries exact match first, then falls back to matching the last
    path component against known model short names.
    """
    if model_name in BACKBONE_IMAGE_TOKEN_IDS:
        return BACKBONE_IMAGE_TOKEN_IDS[model_name]

    # Extract the last component of the path (e.g. "/data1/models/llava-1.5-7b-hf" -> "llava-1.5-7b-hf")
    short_name = model_name.rstrip("/").split("/")[-1]
    if short_name in _BACKBONE_IMAGE_TOKEN_IDS:
        return _BACKBONE_IMAGE_TOKEN_IDS[short_name]

    # Try substring match as last resort
    for key, token_id in _BACKBONE_IMAGE_TOKEN_IDS.items():
        if key in model_name:
            return token_id

    raise KeyError(
        f"Cannot find image token ID for model '{model_name}'. "
        f"Known models: {list(_BACKBONE_IMAGE_TOKEN_IDS.keys())}"
    )

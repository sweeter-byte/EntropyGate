"""EDGE method registration — patches model forward and sampling."""

from edge.sampler import patch_edge_sampling
from edge.model_forward.llama_forward import patch_llama_forward
from edge.model_forward.qwen_forward import patch_qwen_forward


def patch_everything():
    """Apply all EDGE monkey-patches to transformers."""
    patch_edge_sampling()
    patch_llama_forward()
    patch_qwen_forward()

# EDGE: Entropy-Driven Gated Decoding

A training-free, entropy-gated nested contrastive decoding method for hallucination mitigation in large vision-language models (LVLMs).

## Core Formula

```
intermediate = log_p + (1 - γ_t) / γ_t × (log_p - log_p_lang)
final = (1 + g_vis) × intermediate - g_vis × log_p_stat
g_vis = α_min + (α_base - α_min) × σ((H_t - η) / τ)
```

Where:
- `H_t`: normalized entropy of the output distribution at token `t`
- `g_vis`: entropy-gated visual contrast strength (dynamically in `[α_min, α_base]`)
- `γ_t`: exponentially decayed time factor for language prior
- Three forward passes per token: original, language prior, statistical bias

## Best Configuration (E5)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `alpha_base_vis` | 1.5 | Visual gate upper bound |
| `alpha_min_vis` | 0.5 | Visual gate lower bound |
| `eta_vis` | 0.1 | Entropy threshold |
| `tau_gate` | 0.05 | Sigmoid temperature |
| `gamma_decay` | 0.01 | Time decay coefficient |
| `beta_cutoff` | 0.1 | Plausibility cutoff |
| `theta_safe` | 0.99 | Safety skip threshold |

## Results (LLaVA-1.5-7B, 4-bit, CHAIR 500)

| Method | CHAIRs ↓ | CHAIRi ↓ | Recall ↑ |
|--------|----------|----------|----------|
| Vanilla | 52.6 | 16.3 | 72.9 |
| CRoPS | 37.4 | 10.3 | 72.8 |
| **EDGE** | **35.6** | **9.9** | **73.6** |

## Directory Structure

```
edge/
├── __init__.py                    # Package metadata
├── constants.py                   # All hyperparameter constants
├── generation_config.py           # EdgeGenerationConfig
├── sampler.py                     # Core entropy-gated nested sampling
├── method.py                      # Monkey-patch registration
├── run.py                         # Main execution script
├── model_forward/
│   ├── llama_forward.py           # LLaMA forward hook (Fast-V + Text-Mask)
│   └── qwen_forward.py           # Qwen2/2.5-VL forward hook
├── utils/
│   ├── attention_mask.py          # Fast-V and Text-Mask implementation
│   ├── sampler_utils.py           # Forward pass helpers
│   └── reproducibility.py         # Seed and determinism
├── benchmark/
│   ├── chair.py                   # CHAIR benchmark
│   ├── pope.py                    # POPE benchmark
│   ├── amber.py                   # AMBER benchmark
│   └── evaluators/
│       ├── chair_evaluator.py     # CHAIR metric computation
│       └── mme_utils.py           # MME evaluation utilities
└── scripts/
    ├── run_edge.sh                # Main experiment script (all benchmarks)
    ├── run_dmas_comparison.sh     # DMAS paper-aligned comparison
    └── setup_datasets.sh          # Dataset download helper
```

## Quick Start

```bash
# Run CHAIR benchmark with E5 best config
bash edge/scripts/run_edge.sh chair

# Run POPE benchmark
bash edge/scripts/run_edge.sh pope

# Run MME benchmark
bash edge/scripts/run_edge.sh mme

# Run all benchmarks
bash edge/scripts/run_edge.sh all
```

## Supported Models

- LLaVA-1.5-7B / 13B
- LLaVA-NeXT-8B
- Qwen2-VL-7B / Qwen2.5-VL-7B

## Alignment with DMAS (ICLR 2026)

This method is designed to align experiments with [Dynamic Multimodal Activation Steering](https://arxiv.org/abs/2602.21704) (DMAS). The comparison requires:

| Benchmark | DMAS Metrics | Status |
|-----------|-------------|--------|
| **CHAIR** | CHAIRs ↓, CHAIRi ↓ (500 COCO images, max_tokens=512) | Supported |
| **POPE** | Accuracy, Precision, Recall, F1 (MSCOCO + GQA, 3 splits) | Supported |
| **MME** | Accuracy + Accuracy+ (existence, count, position, color) | Supported |
| **AMBER** | CHAIR, HAL, Cog, AMBER Score | Supported |

DMAS baselines: VCD, OPERA, VAF, DECO, DAMO, AGLA, ICT, VTI

Models: LLaVA-v1.5 7B, QwenVL 7B

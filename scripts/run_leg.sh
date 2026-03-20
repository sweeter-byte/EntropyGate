#!/bin/bash
set -euo pipefail

# =============================================================================
# LEG: Latent Entropy Gate
#
# Same nested formula as EntropyGate E5, but the entropy signal that drives
# the gate comes from a MIDDLE hidden layer (projected via norm + lm_head)
# instead of from the output logits.
#
# Mid-layer entropy may detect hallucination risk earlier/differently.
# Sweeps the hidden layer index and gate parameters.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
QUANTIZATION="${QUANTIZATION:-4bit}"
MODEL="${MODEL:-/data1/ranmaoyin/models/llava-1.5-7b-hf}"

COCO_PATH="${COCO_PATH:-/data1/ranmaoyin/dataset/coco2014/annotations}"
COCO_FILE="${COCO_FILE:-instances_val2014.json}"
COCO_BASE_IMAGE_PATH="${COCO_BASE_IMAGE_PATH:-/data1/ranmaoyin/dataset/coco2014/val2014}"
CHAIR_TEST_SIZE="${CHAIR_TEST_SIZE:-500}"

LOG_DIR="${PROJECT_ROOT}/logs"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_ROOT}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/leg_${TIMESTAMP}.log"

# Configs: (leg_hidden_layer, alpha_base_vis, alpha_min_vis, eta_vis, tau_gate)
# For a 32-layer LLaMA-7B model, outputs.hidden_states has 33 elements:
#   -1 = post-norm (same as logit entropy, baseline)
#   -2 = last layer input
#   -8 = layer ~25 output (close to output)
#   -16 = layer ~17 output (middle)
#   -24 = layer ~9 output (early-mid)
#
# LASER-inspired insight: mid-layer hidden states carry different uncertainty
# signals than output logits.  Higher η tests whether mid-layer entropy
# should trigger intervention only for genuinely confused positions.
CONFIGS=(
    # L1: mid-layer (-16), match E5 params [0.5, 1.5]
    "-16 1.5 0.5 0.10 0.05"
    # L2: early-mid (-24), match E5 params
    "-24 1.5 0.5 0.10 0.05"
    # L3: near-output (-8), match E5 params
    "-8 1.5 0.5 0.10 0.05"
    # L4: mid-layer (-16), wider range [0.3, 2.0]
    "-16 2.0 0.3 0.10 0.05"
    # L5: mid-layer (-16), LASER-inspired η=0.30 — mid-layer entropy is
    #     typically higher than output entropy, so a higher threshold is natural
    "-16 1.5 0.3 0.30 0.10"
)

echo "============================================================" | tee -a "${LOG_FILE}"
echo "LEG: Latent Entropy Gate — $(date)" | tee -a "${LOG_FILE}"
echo "Model: ${MODEL}" | tee -a "${LOG_FILE}"
echo "GPU: ${GPU_ID}  |  Quantization: ${QUANTIZATION}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

IDX=0
for CFG in "${CONFIGS[@]}"; do
    IDX=$((IDX + 1))
    read -r HLAYER ABASE AMIN ETA TAU <<< "${CFG}"
    EXPERIMENT_NAME="leg_l${IDX}_${TIMESTAMP}"

    echo "" | tee -a "${LOG_FILE}"
    echo "------------------------------------------------------------" | tee -a "${LOG_FILE}"
    echo "L${IDX}: hidden_layer=${HLAYER} alpha_base=${ABASE} alpha_min=${AMIN} eta=${ETA} tau=${TAU}" | tee -a "${LOG_FILE}"
    echo "------------------------------------------------------------" | tee -a "${LOG_FILE}"

    CMD=(
        python3 "${PROJECT_ROOT}/run_entropygate.py"
        --method latent
        --latent_method leg
        --model_name "${MODEL}"
        --seed "${SEED}"
        --max_new_tokens "${MAX_NEW_TOKENS}"
        --experiment_name "${EXPERIMENT_NAME}"
        --leg_hidden_layer "${HLAYER}"
        --alpha_base_vis "${ABASE}"
        --alpha_min_vis "${AMIN}"
        --eta_vis "${ETA}"
        --tau_gate "${TAU}"
        --gamma_decay 0.01
        --beta_cutoff_fixed 0.1
        --run_chair_benchmark
        --coco_path "${COCO_PATH}"
        --coco_file "${COCO_FILE}"
        --coco_base_image_path "${COCO_BASE_IMAGE_PATH}"
        --chair_test_size "${CHAIR_TEST_SIZE}"
    )

    case "${QUANTIZATION}" in
        none) ;;
        4bit) CMD+=(--load_in_4bit) ;;
        8bit) CMD+=(--load_in_8bit) ;;
    esac

    echo "CMD: ${CMD[*]}" | tee -a "${LOG_FILE}"
    "${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"

    echo "L${IDX} done." | tee -a "${LOG_FILE}"
done

echo "" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "LEG experiments complete. Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

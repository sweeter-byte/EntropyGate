#!/bin/bash
set -euo pipefail

# =============================================================================
# Ablation: Adaptive cutoff β (design doc Section 3.4, 消融四)
# Compares fixed β=0.1 (CRoPS original) vs adaptive β_t (EntropyGate)
# Hardware: NVIDIA L40S
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
QUANTIZATION="${QUANTIZATION:-none}"
MODEL="${MODEL:-llava-hf/llava-1.5-7b-hf}"

COCO_PATH="${COCO_PATH:-${PROJECT_ROOT}/dataset/annotations}"
COCO_FILE="${COCO_FILE:-instances_val2014.json}"
COCO_BASE_IMAGE_PATH="${COCO_BASE_IMAGE_PATH:-${PROJECT_ROOT}/dataset/val2014}"
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
LOG_FILE="${LOG_DIR}/ablation_cutoff_${TIMESTAMP}.log"

# (β_base, β_range) pairs:
#   fixed β=0.1  → beta_base=0.1, beta_range=0.0 (no adaptation)
#   adaptive β_t → beta_base=0.05, beta_range=0.15 (default EntropyGate)
CUTOFF_CONFIGS=(
    "0.1  0.0   fixed_beta_0.1"
    "0.05 0.15  adaptive_beta"
)

echo "============================================================" | tee -a "${LOG_FILE}"
echo "Ablation: Adaptive cutoff β — $(date)" | tee -a "${LOG_FILE}"
echo "Model: ${MODEL}" | tee -a "${LOG_FILE}"
echo "GPU: ${GPU_ID}  |  Quantization: ${QUANTIZATION}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

for CONFIG in "${CUTOFF_CONFIGS[@]}"; do
    read -r B_BASE B_RANGE LABEL <<< "${CONFIG}"
    EXPERIMENT_NAME="ablation_cutoff_${LABEL}_${TIMESTAMP}"

    echo "" | tee -a "${LOG_FILE}"
    echo "------------------------------------------------------------" | tee -a "${LOG_FILE}"
    echo "${LABEL}: beta_base=${B_BASE}  beta_range=${B_RANGE}" | tee -a "${LOG_FILE}"
    echo "------------------------------------------------------------" | tee -a "${LOG_FILE}"

    CMD=(
        python3 "${PROJECT_ROOT}/run_entropygate.py"
        --model_name "${MODEL}"
        --seed "${SEED}"
        --max_new_tokens "${MAX_NEW_TOKENS}"
        --experiment_name "${EXPERIMENT_NAME}"
        --beta_base "${B_BASE}"
        --beta_range "${B_RANGE}"
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

    echo "${LABEL} done." | tee -a "${LOG_FILE}"
done

echo "" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "Cutoff ablation complete. Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

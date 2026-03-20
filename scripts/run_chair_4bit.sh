#!/bin/bash
set -euo pipefail

# =============================================================================
# Step 1: 4bit quantization + 500 samples (same as before, reproducibility check)
# Purpose: Confirm results are stable across runs under 4bit
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID="${GPU_ID:-3}"
SEED="${SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

MODEL_NAME="${MODEL:-/data1/ranmaoyin/models/llava-1.5-7b-hf}"
METHODS=("vanilla" "crops" "entropygate")

COCO_PATH="/data1/ranmaoyin/dataset/coco2014/annotations"
COCO_FILE="instances_val2014.json"
COCO_BASE_IMAGE_PATH="/data1/ranmaoyin/dataset/coco2014/val2014"
CHAIR_TEST_SIZE="${CHAIR_TEST_SIZE:-500}"

LOG_DIR="${PROJECT_ROOT}/logs"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_ROOT}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${LOG_DIR}/chair_fullset_4bit_${TIMESTAMP}.log"

echo "============================================================" | tee -a "${MASTER_LOG}"
echo "CHAIR Full Set (4bit) — $(date)" | tee -a "${MASTER_LOG}"
echo "Model: ${MODEL_NAME}" | tee -a "${MASTER_LOG}"
echo "Methods: ${METHODS[*]}" | tee -a "${MASTER_LOG}"
echo "CHAIR_TEST_SIZE: ${CHAIR_TEST_SIZE}" | tee -a "${MASTER_LOG}"
echo "Quantization: 4bit" | tee -a "${MASTER_LOG}"
echo "GPU: ${GPU_ID}" | tee -a "${MASTER_LOG}"
echo "============================================================" | tee -a "${MASTER_LOG}"

for METHOD in "${METHODS[@]}"; do
    EXPERIMENT_NAME="${METHOD}_chair_fullset_4bit_${TIMESTAMP}"

    echo "" | tee -a "${MASTER_LOG}"
    echo "------------------------------------------------------------" | tee -a "${MASTER_LOG}"
    echo "Method: ${METHOD} | CHAIR full set | 4bit" | tee -a "${MASTER_LOG}"
    echo "------------------------------------------------------------" | tee -a "${MASTER_LOG}"

    CMD=(
        python3 "${PROJECT_ROOT}/run_entropygate.py"
        --method "${METHOD}"
        --model_name "${MODEL_NAME}"
        --seed "${SEED}"
        --max_new_tokens "${MAX_NEW_TOKENS}"
        --experiment_name "${EXPERIMENT_NAME}"
        --load_in_4bit
        --run_chair_benchmark
        --coco_path "${COCO_PATH}"
        --coco_file "${COCO_FILE}"
        --coco_base_image_path "${COCO_BASE_IMAGE_PATH}"
        --chair_test_size "${CHAIR_TEST_SIZE}"
    )

    case "${METHOD}" in
        vanilla)
            CMD+=(--do_sample)
            ;;
        crops)
            CMD+=(
                --do_sample
                --lambda_lang_prior 0.01
                --alpha_stat_bias 1
                --beta_cutoff 0.1
                --max_threshold_plausibility_constraint 0.95
            )
            ;;
        entropygate)
            ;;
    esac

    echo "CMD: ${CMD[*]}" | tee -a "${MASTER_LOG}"
    "${CMD[@]}" 2>&1 | tee -a "${MASTER_LOG}"

    echo "${METHOD}/chair_fullset_4bit done." | tee -a "${MASTER_LOG}"
done

echo "" | tee -a "${MASTER_LOG}"
echo "============================================================" | tee -a "${MASTER_LOG}"
echo "All done. Log: ${MASTER_LOG}" | tee -a "${MASTER_LOG}"
echo "============================================================" | tee -a "${MASTER_LOG}"

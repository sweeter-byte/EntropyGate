#!/bin/bash
set -euo pipefail

# =============================================================================
# Run vanilla / CRoPS / EntropyGate on AMBER generative benchmark
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID="3"
SEED="${SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MODEL="${MODEL:-/data1/ranmaoyin/models/llava-1.5-7b-hf}"
METHODS=("vanilla" "crops" "entropygate")

AMBER_ROOT="${AMBER_ROOT:-/data1/ranmaoyin/dataset/amber}"
AMBER_QUERY_FILE="${AMBER_QUERY_FILE:-${AMBER_ROOT}/data/query/query_generative.json}"
AMBER_IMAGE_DIR="${AMBER_IMAGE_DIR:-${AMBER_ROOT}/images}"
AMBER_OFFICIAL_REPO_PATH="${AMBER_OFFICIAL_REPO_PATH:-${AMBER_ROOT}/official_repo}"
AMBER_EVALUATION_TYPE="${AMBER_EVALUATION_TYPE:-g}"

LOG_DIR="${PROJECT_ROOT}/logs"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_ROOT}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${LOG_DIR}/amber_experiments_${TIMESTAMP}.log"

echo "============================================================" | tee -a "${MASTER_LOG}"
echo "AMBER Experiments — $(date)" | tee -a "${MASTER_LOG}"
echo "Model: ${MODEL}" | tee -a "${MASTER_LOG}"
echo "Methods: ${METHODS[*]}" | tee -a "${MASTER_LOG}"
echo "GPU: ${GPU_ID}" | tee -a "${MASTER_LOG}"
echo "AMBER query: ${AMBER_QUERY_FILE}" | tee -a "${MASTER_LOG}"
echo "AMBER images: ${AMBER_IMAGE_DIR}" | tee -a "${MASTER_LOG}"
echo "============================================================" | tee -a "${MASTER_LOG}"

for METHOD in "${METHODS[@]}"; do
    EXPERIMENT_NAME="${METHOD}_amber_${TIMESTAMP}"

    echo "" | tee -a "${MASTER_LOG}"
    echo "------------------------------------------------------------" | tee -a "${MASTER_LOG}"
    echo "Method: ${METHOD}" | tee -a "${MASTER_LOG}"
    echo "------------------------------------------------------------" | tee -a "${MASTER_LOG}"

    CMD=(
        python3 "${PROJECT_ROOT}/run_entropygate.py"
        --method "${METHOD}"
        --model_name "${MODEL}"
        --seed "${SEED}"
        --max_new_tokens "${MAX_NEW_TOKENS}"
        --experiment_name "${EXPERIMENT_NAME}"
        --load_in_4bit
        --run_amber_benchmark
        --amber_query_file "${AMBER_QUERY_FILE}"
        --amber_image_dir "${AMBER_IMAGE_DIR}"
        --amber_official_repo_path "${AMBER_OFFICIAL_REPO_PATH}"
        --amber_evaluation_type "${AMBER_EVALUATION_TYPE}"
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

    echo "${METHOD}/amber done." | tee -a "${MASTER_LOG}"
done

echo "" | tee -a "${MASTER_LOG}"
echo "============================================================" | tee -a "${MASTER_LOG}"
echo "AMBER experiments complete. Master log: ${MASTER_LOG}" | tee -a "${MASTER_LOG}"
echo "============================================================" | tee -a "${MASTER_LOG}"

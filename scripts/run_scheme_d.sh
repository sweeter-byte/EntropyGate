#!/bin/bash
set -euo pipefail

# =============================================================================
# Scheme D: Floor-gated EntropyGate
# g_vis = alpha_min_vis + (alpha_base_vis - alpha_min_vis) * sigmoid(...)
# Sweeps alpha_min_vis values to find the best floor for visual contrast.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID="3"
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
LOG_FILE="${LOG_DIR}/scheme_d_floor_${TIMESTAMP}.log"

# (alpha_min_vis, alpha_min_txt) pairs to sweep
# alpha_base_vis=1.5, alpha_base_txt=1.0 (ceiling); floor varies
ALPHA_BASE_VIS="1.5"
ALPHA_BASE_TXT="1.0"

FLOOR_PAIRS=(
    "0.0 0.0"    # baseline: no floor (original EntropyGate)
    "0.3 0.1"    # light floor
    "0.5 0.2"    # moderate floor
    "0.7 0.3"    # strong floor
    "1.0 0.5"    # very strong floor (g_vis always >= 1.0, like CRoPS)
)

echo "============================================================" | tee -a "${LOG_FILE}"
echo "Scheme D: Floor-gated EntropyGate — $(date)" | tee -a "${LOG_FILE}"
echo "Model: ${MODEL}" | tee -a "${LOG_FILE}"
echo "alpha_base_vis=${ALPHA_BASE_VIS} alpha_base_txt=${ALPHA_BASE_TXT}" | tee -a "${LOG_FILE}"
echo "GPU: ${GPU_ID}  |  Quantization: ${QUANTIZATION}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

for PAIR in "${FLOOR_PAIRS[@]}"; do
    read -r AMIN_VIS AMIN_TXT <<< "${PAIR}"
    EXPERIMENT_NAME="scheme_d_floor_vis${AMIN_VIS}_txt${AMIN_TXT}_${TIMESTAMP}"

    echo "" | tee -a "${LOG_FILE}"
    echo "------------------------------------------------------------" | tee -a "${LOG_FILE}"
    echo "alpha_min_vis=${AMIN_VIS}  alpha_min_txt=${AMIN_TXT}" | tee -a "${LOG_FILE}"
    echo "------------------------------------------------------------" | tee -a "${LOG_FILE}"

    CMD=(
        python3 "${PROJECT_ROOT}/run_entropygate.py"
        --model_name "${MODEL}"
        --seed "${SEED}"
        --max_new_tokens "${MAX_NEW_TOKENS}"
        --experiment_name "${EXPERIMENT_NAME}"
        --alpha_base_vis "${ALPHA_BASE_VIS}"
        --alpha_base_txt "${ALPHA_BASE_TXT}"
        --alpha_min_vis "${AMIN_VIS}"
        --alpha_min_txt "${AMIN_TXT}"
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

    echo "alpha_min_vis=${AMIN_VIS} alpha_min_txt=${AMIN_TXT} done." | tee -a "${LOG_FILE}"
done

echo "" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "Scheme D complete. Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

#!/bin/bash
set -euo pipefail

# =============================================================================
# Scheme F: ACD-style entropy ratio gating + nested structure
#
# Formula:
#   alpha_vis = H_orig / (H_orig + H_stat)   (auto-adaptive)
#   alpha_txt = H_orig / (H_orig + H_lang)   (auto-adaptive)
#   intermediate = log_p + alpha_txt * (1-gamma_t)/gamma_t * (log_p - log_p_lang)
#   final = (1 + alpha_vis) * intermediate - alpha_vis * log_p_stat
#
# No eta/tau needed — gate strength is fully determined by entropy ratios.
# Sweeps alpha_min_vis (floor) to prevent under-correction.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID="${GPU_ID:-3}"
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
LOG_FILE="${LOG_DIR}/scheme_f_${TIMESTAMP}.log"

# Configs: (alpha_min_vis, alpha_min_txt)
# ACD ratio is auto-computed; floor prevents under-correction
CONFIGS=(
    # F1: no floor — pure ACD ratio
    "0.0 0.0"
    # F2: light floor
    "0.3 0.1"
    # F3: moderate floor (recommended)
    "0.5 0.2"
    # F4: strong floor
    "0.7 0.3"
)

echo "============================================================" | tee -a "${LOG_FILE}"
echo "Scheme F: ACD Entropy Ratio + Nested — $(date)" | tee -a "${LOG_FILE}"
echo "Model: ${MODEL}" | tee -a "${LOG_FILE}"
echo "GPU: ${GPU_ID}  |  Quantization: ${QUANTIZATION}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

IDX=0
for CFG in "${CONFIGS[@]}"; do
    IDX=$((IDX + 1))
    read -r AMIN_VIS AMIN_TXT <<< "${CFG}"
    EXPERIMENT_NAME="scheme_f_f${IDX}_${TIMESTAMP}"

    echo "" | tee -a "${LOG_FILE}"
    echo "------------------------------------------------------------" | tee -a "${LOG_FILE}"
    echo "F${IDX}: alpha_min_vis=${AMIN_VIS} alpha_min_txt=${AMIN_TXT}" | tee -a "${LOG_FILE}"
    echo "------------------------------------------------------------" | tee -a "${LOG_FILE}"

    CMD=(
        python3 "${PROJECT_ROOT}/run_entropygate.py"
        --method entropygate
        --model_name "${MODEL}"
        --seed "${SEED}"
        --max_new_tokens "${MAX_NEW_TOKENS}"
        --experiment_name "${EXPERIMENT_NAME}"
        --eg_scheme acd
        --alpha_min_vis "${AMIN_VIS}"
        --alpha_min_txt "${AMIN_TXT}"
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

    echo "F${IDX} done." | tee -a "${LOG_FILE}"
done

echo "" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "Scheme F complete. Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

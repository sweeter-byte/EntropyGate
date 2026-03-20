#!/bin/bash
set -euo pipefail

# =============================================================================
# 方向 1+2: 精细校准 eta/tau + 解耦 time_decay
#
# 保持原始平坦公式不变，只调参数:
#   方向1: sweep eta ∈ {0.10, 0.12} × tau ∈ {0.02, 0.03} × alpha_base_txt ∈ {2.0, 3.0}
#   方向2: time_decay_mode=additive, alpha_time_txt sweep
#
# 所有配置都用 floor: alpha_min_vis=0.5, alpha_min_txt=0.2 (D+A c1 最优)
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
LOG_FILE="${LOG_DIR}/dir12_${TIMESTAMP}.log"

# Fixed params (from D+A c1 best)
AMIN_VIS="0.5"
AMIN_TXT="0.2"
ABASE_VIS="1.5"

# Configs: (label, eta_vis, eta_txt, tau, alpha_base_txt, time_decay_mode, alpha_time_txt)
CONFIGS=(
    # ---- 方向 1: 校准 eta + tau ----
    # D+A c1 baseline for comparison
    "d1_baseline 0.10 0.15 0.05 1.0 multiply 1.0"
    # Sharper tau
    "d1_tau02 0.10 0.15 0.02 1.0 multiply 1.0"
    "d1_tau03 0.10 0.15 0.03 1.0 multiply 1.0"
    # Higher eta (at median)
    "d1_eta12 0.12 0.18 0.02 1.0 multiply 1.0"
    # Larger alpha_base_txt to boost g_txt
    "d1_abtxt2 0.10 0.15 0.02 2.0 multiply 1.0"
    "d1_abtxt3 0.10 0.15 0.02 3.0 multiply 1.0"
    # Best tau + larger alpha_base_txt
    "d1_combo 0.12 0.18 0.03 2.0 multiply 1.0"

    # ---- 方向 2: 解耦 time_decay ----
    # Additive mode: g_txt = base_gate + alpha_time_txt * time_decay
    "d2_add1 0.10 0.15 0.02 1.0 additive 1.0"
    "d2_add2 0.10 0.15 0.02 1.0 additive 2.0"
    "d2_add05 0.10 0.15 0.02 1.0 additive 0.5"
)

echo "============================================================" | tee -a "${LOG_FILE}"
echo "Direction 1+2: Calibrate eta/tau + Decouple time_decay — $(date)" | tee -a "${LOG_FILE}"
echo "Model: ${MODEL}" | tee -a "${LOG_FILE}"
echo "Fixed: alpha_min_vis=${AMIN_VIS} alpha_min_txt=${AMIN_TXT} alpha_base_vis=${ABASE_VIS}" | tee -a "${LOG_FILE}"
echo "GPU: ${GPU_ID}  |  Quantization: ${QUANTIZATION}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

for CFG in "${CONFIGS[@]}"; do
    read -r LABEL ETA_V ETA_T TAU ABASE_TXT TD_MODE A_TIME <<< "${CFG}"
    EXPERIMENT_NAME="dir12_${LABEL}_${TIMESTAMP}"

    echo "" | tee -a "${LOG_FILE}"
    echo "------------------------------------------------------------" | tee -a "${LOG_FILE}"
    echo "${LABEL}: eta_vis=${ETA_V} eta_txt=${ETA_T} tau=${TAU} alpha_base_txt=${ABASE_TXT} td_mode=${TD_MODE} alpha_time=${A_TIME}" | tee -a "${LOG_FILE}"
    echo "------------------------------------------------------------" | tee -a "${LOG_FILE}"

    CMD=(
        python3 "${PROJECT_ROOT}/run_entropygate.py"
        --method entropygate
        --model_name "${MODEL}"
        --seed "${SEED}"
        --max_new_tokens "${MAX_NEW_TOKENS}"
        --experiment_name "${EXPERIMENT_NAME}"
        --eg_scheme flat
        --alpha_base_vis "${ABASE_VIS}"
        --alpha_base_txt "${ABASE_TXT}"
        --alpha_min_vis "${AMIN_VIS}"
        --alpha_min_txt "${AMIN_TXT}"
        --eta_vis "${ETA_V}"
        --eta_txt "${ETA_T}"
        --tau_gate "${TAU}"
        --gamma_decay 0.01
        --time_decay_mode "${TD_MODE}"
        --alpha_time_txt "${A_TIME}"
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
echo "Direction 1+2 complete. Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

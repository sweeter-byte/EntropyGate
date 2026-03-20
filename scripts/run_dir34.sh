#!/bin/bash
set -euo pipefail

# =============================================================================
# 方向 3+4: 自适应 eta + 选择性激活
#
# 方向3: adaptive_eta=True, eta 自动跟踪运行时熵分布
# 方向4: soft_suppress=True, 用 (1-max_prob^k) 平滑衰减对比强度
#
# 基于方向1+2的最优参数运行（先用 D+A c1 参数作为 baseline）
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
LOG_FILE="${LOG_DIR}/dir34_${TIMESTAMP}.log"

# Fixed params
AMIN_VIS="0.5"
AMIN_TXT="0.2"
ABASE_VIS="1.5"
TAU="0.02"

echo "============================================================" | tee -a "${LOG_FILE}"
echo "Direction 3+4: Adaptive Eta + Soft Suppress — $(date)" | tee -a "${LOG_FILE}"
echo "Model: ${MODEL}" | tee -a "${LOG_FILE}"
echo "GPU: ${GPU_ID}  |  Quantization: ${QUANTIZATION}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

run_experiment() {
    local LABEL="$1"
    local ABASE_TXT="$2"
    local ETA_V="$3"
    local ETA_T="$4"
    local EXTRA_ARGS=("${@:5}")

    local EXPERIMENT_NAME="dir34_${LABEL}_${TIMESTAMP}"

    echo "" | tee -a "${LOG_FILE}"
    echo "------------------------------------------------------------" | tee -a "${LOG_FILE}"
    echo "${LABEL}: ${EXTRA_ARGS[*]}" | tee -a "${LOG_FILE}"
    echo "------------------------------------------------------------" | tee -a "${LOG_FILE}"

    local CMD=(
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
        "${EXTRA_ARGS[@]}"
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
}

# ---- 方向 3: 自适应 eta ----
# eta_vis/eta_txt are initial values (will be overridden by EMA)
# eta_vis_offset/eta_txt_offset control: eta = H_mean + offset * H_std

# 3a: conservative offset (eta slightly above mean)
run_experiment "d3_off05_10" "1.0" "0.12" "0.18" \
    --adaptive_eta --eta_vis_offset 0.5 --eta_txt_offset 1.0

# 3b: aggressive offset (eta near mean → gate opens more)
run_experiment "d3_off0_05" "1.0" "0.12" "0.18" \
    --adaptive_eta --eta_vis_offset 0.0 --eta_txt_offset 0.5

# 3c: adaptive eta + larger alpha_base_txt
run_experiment "d3_abtxt2" "2.0" "0.12" "0.18" \
    --adaptive_eta --eta_vis_offset 0.5 --eta_txt_offset 1.0

# 3d: adaptive eta + faster EMA (more responsive)
run_experiment "d3_ema02" "1.0" "0.12" "0.18" \
    --adaptive_eta --eta_ema_momentum 0.2 --eta_vis_offset 0.5 --eta_txt_offset 1.0

# ---- 方向 4: 选择性激活 (soft suppress) ----
# suppress_factor = 1 - max_prob^k

# 4a: k=4 (moderate suppression)
run_experiment "d4_k4" "1.0" "0.10" "0.15" \
    --soft_suppress --soft_suppress_k 4.0

# 4b: k=8 (sharper — only suppress when very confident)
run_experiment "d4_k8" "1.0" "0.10" "0.15" \
    --soft_suppress --soft_suppress_k 8.0

# 4c: k=2 (gentle — suppress more broadly)
run_experiment "d4_k2" "1.0" "0.10" "0.15" \
    --soft_suppress --soft_suppress_k 2.0

# ---- 组合: 方向 3+4 ----
# 自适应 eta + soft suppress
run_experiment "d34_combo" "2.0" "0.12" "0.18" \
    --adaptive_eta --eta_vis_offset 0.5 --eta_txt_offset 1.0 \
    --soft_suppress --soft_suppress_k 4.0

echo "" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "Direction 3+4 complete. Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

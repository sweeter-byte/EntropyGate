#!/bin/bash
set -euo pipefail

# =============================================================================
# Scheme D+A: Floor-gated EntropyGate with adaptive eta
# - Floor: alpha_min_vis/txt guarantees minimum contrastive strength
# - Adaptive eta: lower thresholds calibrated to actual entropy distribution
# Sweeps (alpha_min_vis, eta_vis, eta_txt) combinations
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
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_ROOT}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/scheme_d_a_${TIMESTAMP}.log"

# Based on log analysis: avg_H ≈ 0.10~0.15 for LLaVA-1.5-7B
# Scheme A: set eta near the median of actual H distribution
# Scheme D: set floor to guarantee baseline contrastive strength
#
# Configs: (alpha_min_vis, alpha_min_txt, eta_vis, eta_txt, tau_gate, alpha_base_vis)
CONFIGS=(
    # D+A config 1: moderate floor + eta at P40 of H distribution
    "0.5 0.2 0.10 0.15 0.05 1.5"
    # D+A config 2: moderate floor + eta at P50
    "0.5 0.2 0.12 0.18 0.05 1.5"
    # D+A config 3: strong floor + low eta (aggressive contrastive)
    "0.7 0.3 0.08 0.12 0.05 1.5"
    # D+A config 4: moderate floor + softer sigmoid for smoother gating
    "0.5 0.2 0.10 0.15 0.10 1.5"
    # D+A config 5: strong floor + eta at P50 + softer sigmoid
    "0.7 0.3 0.12 0.18 0.10 1.5"
    # D+A config 6: CRoPS-equivalent floor, entropy only boosts beyond CRoPS
    "1.0 0.5 0.12 0.18 0.05 1.5"
)

echo "============================================================" | tee -a "${LOG_FILE}"
echo "Scheme D+A: Floor + Adaptive Eta — $(date)" | tee -a "${LOG_FILE}"
echo "Model: ${MODEL}" | tee -a "${LOG_FILE}"
echo "GPU: ${GPU_ID}  |  Quantization: ${QUANTIZATION}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

IDX=0
for CFG in "${CONFIGS[@]}"; do
    IDX=$((IDX + 1))
    read -r AMIN_VIS AMIN_TXT ETA_VIS ETA_TXT TAU ABASE_VIS <<< "${CFG}"
    EXPERIMENT_NAME="scheme_da_c${IDX}_${TIMESTAMP}"

    echo "" | tee -a "${LOG_FILE}"
    echo "------------------------------------------------------------" | tee -a "${LOG_FILE}"
    echo "Config ${IDX}: alpha_min_vis=${AMIN_VIS} alpha_min_txt=${AMIN_TXT} eta_vis=${ETA_VIS} eta_txt=${ETA_TXT} tau=${TAU} alpha_base_vis=${ABASE_VIS}" | tee -a "${LOG_FILE}"
    echo "------------------------------------------------------------" | tee -a "${LOG_FILE}"

    CMD=(
        python3 "${PROJECT_ROOT}/run_entropygate.py"
        --model_name "${MODEL}"
        --seed "${SEED}"
        --max_new_tokens "${MAX_NEW_TOKENS}"
        --experiment_name "${EXPERIMENT_NAME}"
        --alpha_base_vis "${ABASE_VIS}"
        --alpha_min_vis "${AMIN_VIS}"
        --alpha_min_txt "${AMIN_TXT}"
        --eta_vis "${ETA_VIS}"
        --eta_txt "${ETA_TXT}"
        --tau_gate "${TAU}"
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

    echo "Config ${IDX} done." | tee -a "${LOG_FILE}"
done

echo "" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "Scheme D+A complete. Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

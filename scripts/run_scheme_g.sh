#!/bin/bash
set -euo pipefail

# =============================================================================
# Scheme G: Scheme E (nested + entropy gate) with aligned CRoPS parameters
#
# Same nested formula as Scheme E, but with:
#   - theta_safe = 0.95 (aligned with CRoPS max_threshold_plausibility_constraint)
#   - beta_cutoff = 0.1 (fixed, aligned with CRoPS)
#
# This isolates the formula structure as the only variable vs CRoPS.
# Uses the best configs from Scheme E.
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
LOG_FILE="${LOG_DIR}/scheme_g_${TIMESTAMP}.log"

# Configs: (alpha_min_vis, alpha_base_vis, eta_vis, tau_gate)
# theta_safe_aligned=0.95, beta_cutoff_fixed=0.1 (CRoPS defaults)
CONFIGS=(
    # G1: g_vis in [0.5, 1.0] + CRoPS-aligned params
    "0.5 1.0 0.10 0.05"
    # G2: g_vis in [0.5, 1.2] + CRoPS-aligned params
    "0.5 1.2 0.10 0.05"
    # G3: g_vis in [0.8, 1.2] + CRoPS-aligned params
    "0.8 1.2 0.10 0.05"
    # G4: g_vis fixed 1.0 + CRoPS-aligned (should ≈ CRoPS exactly)
    "1.0 1.0 0.10 0.05"
    # G5: g_vis in [0.5, 1.5] + CRoPS-aligned params
    "0.5 1.5 0.10 0.05"
)

echo "============================================================" | tee -a "${LOG_FILE}"
echo "Scheme G: Nested + Aligned Params — $(date)" | tee -a "${LOG_FILE}"
echo "Model: ${MODEL}" | tee -a "${LOG_FILE}"
echo "theta_safe_aligned=0.95  beta_cutoff_fixed=0.1" | tee -a "${LOG_FILE}"
echo "GPU: ${GPU_ID}  |  Quantization: ${QUANTIZATION}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

IDX=0
for CFG in "${CONFIGS[@]}"; do
    IDX=$((IDX + 1))
    read -r AMIN_VIS ABASE_VIS ETA_VIS TAU <<< "${CFG}"
    EXPERIMENT_NAME="scheme_g_g${IDX}_${TIMESTAMP}"

    echo "" | tee -a "${LOG_FILE}"
    echo "------------------------------------------------------------" | tee -a "${LOG_FILE}"
    echo "G${IDX}: alpha_min_vis=${AMIN_VIS} alpha_base_vis=${ABASE_VIS} eta_vis=${ETA_VIS} tau=${TAU}" | tee -a "${LOG_FILE}"
    echo "------------------------------------------------------------" | tee -a "${LOG_FILE}"

    CMD=(
        python3 "${PROJECT_ROOT}/run_entropygate.py"
        --method entropygate
        --model_name "${MODEL}"
        --seed "${SEED}"
        --max_new_tokens "${MAX_NEW_TOKENS}"
        --experiment_name "${EXPERIMENT_NAME}"
        --eg_scheme nested_aligned
        --alpha_base_vis "${ABASE_VIS}"
        --alpha_min_vis "${AMIN_VIS}"
        --eta_vis "${ETA_VIS}"
        --tau_gate "${TAU}"
        --gamma_decay 0.01
        --beta_cutoff_fixed 0.1
        --theta_safe_aligned 0.95
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

    echo "G${IDX} done." | tee -a "${LOG_FILE}"
done

echo "" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "Scheme G complete. Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

#!/bin/bash
set -euo pipefail

# =============================================================================
# DMAS-aligned Comparison Experiments for EDGE
#
# Reproduces the experimental setup from:
#   "Dynamic Multimodal Activation Steering for Hallucination Mitigation
#    in Large Vision-Language Models" (ICLR 2026)
#
# Experimental plan:
#   Models: LLaVA-v1.5-7B, Qwen2.5-VL-7B
#   Benchmarks: CHAIR, POPE (MSCOCO 3 splits), MME (4 subtasks)
#   Settings: temperature=0, top_p=1 (greedy), max_new_tokens=512
#
# DMAS results to beat:
#   LLaVA-v1.5 CHAIR: CHAIRs=30.8, CHAIRi=11.4
#   LLaVA-v1.5 POPE MSCOCO avg: Acc=86.81, F1=86.79
#   LLaVA-v1.5 MME Total: 659.99
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"
QUANTIZATION="${QUANTIZATION:-4bit}"

# Model paths (adjust to your environment)
LLAVA_MODEL="${LLAVA_MODEL:-/data1/ranmaoyin/models/llava-1.5-7b-hf}"
QWEN_MODEL="${QWEN_MODEL:-/data1/ranmaoyin/models/Qwen2.5-VL-7B-Instruct}"

# Dataset paths
COCO_PATH="${COCO_PATH:-/data1/ranmaoyin/dataset/coco2014/annotations}"
COCO_FILE="${COCO_FILE:-instances_val2014.json}"
COCO_BASE_IMAGE_PATH="${COCO_BASE_IMAGE_PATH:-/data1/ranmaoyin/dataset/coco2014/val2014}"
POPE_PATH="${POPE_PATH:-/data1/ranmaoyin/dataset/pope}"
POPE_COCO_IMAGE_DIR="${POPE_COCO_IMAGE_DIR:-/data1/ranmaoyin/dataset/coco2014/val2014}"

LOG_DIR="${SCRIPT_DIR}/../logs"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_ROOT}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/dmas_comparison_${TIMESTAMP}.log"

# EDGE hyperparameters (E5 best config)
ALPHA_BASE_VIS=1.5
ALPHA_MIN_VIS=0.5
ETA_VIS=0.10
TAU_GATE=0.05
GAMMA_DECAY=0.01
BETA_CUTOFF=0.1
THETA_SAFE=0.99

# DMAS-aligned generation settings: greedy decoding
MAX_NEW_TOKENS=512

echo "============================================================" | tee -a "${LOG_FILE}"
echo "EDGE vs DMAS Comparison — $(date)" | tee -a "${LOG_FILE}"
echo "Models: LLaVA-v1.5-7B, Qwen2.5-VL-7B" | tee -a "${LOG_FILE}"
echo "Benchmarks: CHAIR, POPE, MME" | tee -a "${LOG_FILE}"
echo "GPU: ${GPU_ID}  |  Quantization: ${QUANTIZATION}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

run_model() {
    local MODEL_PATH="$1"
    local MODEL_TAG="$2"

    echo "" | tee -a "${LOG_FILE}"
    echo "============================================================" | tee -a "${LOG_FILE}"
    echo ">>> Model: ${MODEL_TAG} (${MODEL_PATH})" | tee -a "${LOG_FILE}"
    echo "============================================================" | tee -a "${LOG_FILE}"

    BASE_CMD=(
        python3 "${SCRIPT_DIR}/../run.py"
        --model_name "${MODEL_PATH}"
        --seed "${SEED}"
        --max_new_tokens "${MAX_NEW_TOKENS}"
        --alpha_base_vis "${ALPHA_BASE_VIS}"
        --alpha_min_vis "${ALPHA_MIN_VIS}"
        --eta_vis "${ETA_VIS}"
        --tau_gate "${TAU_GATE}"
        --gamma_decay "${GAMMA_DECAY}"
        --beta_cutoff "${BETA_CUTOFF}"
        --theta_safe "${THETA_SAFE}"
    )

    case "${QUANTIZATION}" in
        none) ;;
        4bit) BASE_CMD+=(--load_in_4bit) ;;
        8bit) BASE_CMD+=(--load_in_8bit) ;;
    esac

    # ----- CHAIR (DMAS Table 3) -----
    echo "" | tee -a "${LOG_FILE}"
    echo ">>> [${MODEL_TAG}] CHAIR benchmark (500 images, max_tokens=512)..." | tee -a "${LOG_FILE}"
    CMD=("${BASE_CMD[@]}"
        --experiment_name "dmas_${MODEL_TAG}_chair_${TIMESTAMP}"
        --run_chair_benchmark
        --coco_path "${COCO_PATH}"
        --coco_file "${COCO_FILE}"
        --coco_base_image_path "${COCO_BASE_IMAGE_PATH}"
        --chair_test_size 500
    )
    echo "CMD: ${CMD[*]}" | tee -a "${LOG_FILE}"
    "${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
    echo ">>> [${MODEL_TAG}] CHAIR done." | tee -a "${LOG_FILE}"

    # ----- POPE MSCOCO (DMAS Table 2, 3 splits) -----
    echo "" | tee -a "${LOG_FILE}"
    echo ">>> [${MODEL_TAG}] POPE benchmark (MSCOCO, 3 splits)..." | tee -a "${LOG_FILE}"
    CMD=("${BASE_CMD[@]}"
        --experiment_name "dmas_${MODEL_TAG}_pope_mscoco_${TIMESTAMP}"
        --run_pope_benchmark
        --pope_path "${POPE_PATH}"
        --pope_coco_image_dir "${POPE_COCO_IMAGE_DIR}"
        --pope_splits random popular adversarial
    )
    echo "CMD: ${CMD[*]}" | tee -a "${LOG_FILE}"
    "${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
    echo ">>> [${MODEL_TAG}] POPE done." | tee -a "${LOG_FILE}"

    # ----- MME (DMAS Table 1) -----
    echo "" | tee -a "${LOG_FILE}"
    echo ">>> [${MODEL_TAG}] MME benchmark (existence/count/position/color)..." | tee -a "${LOG_FILE}"
    CMD=("${BASE_CMD[@]}"
        --experiment_name "dmas_${MODEL_TAG}_mme_${TIMESTAMP}"
        --run_mme_benchmark
    )
    echo "CMD: ${CMD[*]}" | tee -a "${LOG_FILE}"
    "${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
    echo ">>> [${MODEL_TAG}] MME done." | tee -a "${LOG_FILE}"
}

# ===========================================================
# Run experiments
# ===========================================================

# Select which models to run (default: both)
TARGET="${1:-both}"

if [[ "${TARGET}" == "both" || "${TARGET}" == "llava" ]]; then
    if [[ -d "${LLAVA_MODEL}" || "${LLAVA_MODEL}" == llava-hf/* ]]; then
        run_model "${LLAVA_MODEL}" "llava15_7b"
    else
        echo "WARNING: LLaVA model not found at ${LLAVA_MODEL}" | tee -a "${LOG_FILE}"
        echo "  Set LLAVA_MODEL env var or download with: edge/scripts/setup_datasets.sh" | tee -a "${LOG_FILE}"
    fi
fi

if [[ "${TARGET}" == "both" || "${TARGET}" == "qwen" ]]; then
    if [[ -d "${QWEN_MODEL}" || "${QWEN_MODEL}" == Qwen/* ]]; then
        run_model "${QWEN_MODEL}" "qwen25vl_7b"
    else
        echo "WARNING: Qwen model not found at ${QWEN_MODEL}" | tee -a "${LOG_FILE}"
        echo "  Set QWEN_MODEL env var or download with: edge/scripts/setup_datasets.sh" | tee -a "${LOG_FILE}"
    fi
fi

echo "" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "DMAS comparison complete. Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "DMAS reference results (to compare against):" | tee -a "${LOG_FILE}"
echo "  LLaVA-v1.5 CHAIR:  CHAIRs=30.8  CHAIRi=11.4" | tee -a "${LOG_FILE}"
echo "  LLaVA-v1.5 POPE:   Acc=86.81    F1=86.79" | tee -a "${LOG_FILE}"
echo "  LLaVA-v1.5 MME:    Total=659.99" | tee -a "${LOG_FILE}"
echo "  QwenVL POPE:        Acc=87.63    F1=87.65" | tee -a "${LOG_FILE}"
echo "  QwenVL MME:         Total=633.33" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

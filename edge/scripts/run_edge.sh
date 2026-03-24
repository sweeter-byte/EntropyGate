#!/bin/bash
set -euo pipefail

# =============================================================================
# EDGE: Entropy-Driven Gated Decoding — Main Experiment Script
#
# Runs EDGE (E5 configuration) on all three core benchmarks:
#   - CHAIR: open-ended generation hallucination (CHAIRs, CHAIRi, Recall)
#   - POPE:  discriminative hallucination (Accuracy, Precision, Recall, F1)
#   - MME:   VLM perception & cognition (existence, count, position, color)
#
# Best hyperparameters (E5):
#   alpha_base_vis=1.5, alpha_min_vis=0.5, eta_vis=0.10, tau=0.05
#   gamma_decay=0.01, beta_cutoff=0.1, theta_safe=0.99
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
QUANTIZATION="${QUANTIZATION:-4bit}"
MODEL="${MODEL:-llava-hf/llava-1.5-7b-hf}"

# Dataset paths (adjust to your environment)
COCO_PATH="${COCO_PATH:-/data1/ranmaoyin/dataset/coco2014/annotations}"
COCO_FILE="${COCO_FILE:-instances_val2014.json}"
COCO_BASE_IMAGE_PATH="${COCO_BASE_IMAGE_PATH:-/data1/ranmaoyin/dataset/coco2014/val2014}"
CHAIR_TEST_SIZE="${CHAIR_TEST_SIZE:-500}"

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
LOG_FILE="${LOG_DIR}/edge_${TIMESTAMP}.log"

# EDGE hyperparameters (E5 best config)
ALPHA_BASE_VIS=1.5
ALPHA_MIN_VIS=0.5
ETA_VIS=0.10
TAU_GATE=0.05
GAMMA_DECAY=0.01
BETA_CUTOFF=0.1
THETA_SAFE=0.99

BASE_CMD=(
    python3 "${SCRIPT_DIR}/../run.py"
    --model_name "${MODEL}"
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

echo "============================================================" | tee -a "${LOG_FILE}"
echo "EDGE: Entropy-Driven Gated Decoding — $(date)" | tee -a "${LOG_FILE}"
echo "Model: ${MODEL}" | tee -a "${LOG_FILE}"
echo "GPU: ${GPU_ID}  |  Quantization: ${QUANTIZATION}" | tee -a "${LOG_FILE}"
echo "Params: alpha_vis=[${ALPHA_MIN_VIS},${ALPHA_BASE_VIS}] eta=${ETA_VIS} tau=${TAU_GATE}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

# ----- CHAIR Benchmark -----
BENCHMARK="${1:-all}"
if [[ "${BENCHMARK}" == "all" || "${BENCHMARK}" == "chair" ]]; then
    echo "" | tee -a "${LOG_FILE}"
    echo ">>> Running CHAIR benchmark..." | tee -a "${LOG_FILE}"
    CMD=("${BASE_CMD[@]}"
        --experiment_name "edge_chair_${TIMESTAMP}"
        --run_chair_benchmark
        --coco_path "${COCO_PATH}"
        --coco_file "${COCO_FILE}"
        --coco_base_image_path "${COCO_BASE_IMAGE_PATH}"
        --chair_test_size "${CHAIR_TEST_SIZE}"
    )
    echo "CMD: ${CMD[*]}" | tee -a "${LOG_FILE}"
    "${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
    echo ">>> CHAIR done." | tee -a "${LOG_FILE}"
fi

# ----- POPE Benchmark -----
if [[ "${BENCHMARK}" == "all" || "${BENCHMARK}" == "pope" ]]; then
    echo "" | tee -a "${LOG_FILE}"
    echo ">>> Running POPE benchmark..." | tee -a "${LOG_FILE}"
    CMD=("${BASE_CMD[@]}"
        --experiment_name "edge_pope_${TIMESTAMP}"
        --run_pope_benchmark
        --pope_path "${POPE_PATH}"
        --pope_coco_image_dir "${POPE_COCO_IMAGE_DIR}"
        --pope_splits random popular adversarial
    )
    echo "CMD: ${CMD[*]}" | tee -a "${LOG_FILE}"
    "${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
    echo ">>> POPE done." | tee -a "${LOG_FILE}"
fi

# ----- MME Benchmark -----
if [[ "${BENCHMARK}" == "all" || "${BENCHMARK}" == "mme" ]]; then
    echo "" | tee -a "${LOG_FILE}"
    echo ">>> Running MME benchmark..." | tee -a "${LOG_FILE}"
    CMD=("${BASE_CMD[@]}"
        --experiment_name "edge_mme_${TIMESTAMP}"
        --run_mme_benchmark
    )
    echo "CMD: ${CMD[*]}" | tee -a "${LOG_FILE}"
    "${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
    echo ">>> MME done." | tee -a "${LOG_FILE}"
fi

echo "" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "EDGE complete. Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

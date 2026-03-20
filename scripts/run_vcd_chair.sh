#!/bin/bash
set -euo pipefail

# =============================================================================
# VCD vs VCD+EntropyGate on CHAIR (4bit, 500 samples)
# Compares: vanilla / VCD (original) / VCD+EntropyGate
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID="${GPU_ID:-3}"
SEED="${SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
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
MASTER_LOG="${LOG_DIR}/vcd_chair_${TIMESTAMP}.log"

echo "============================================================" | tee -a "${MASTER_LOG}"
echo "VCD CHAIR Experiments — $(date)" | tee -a "${MASTER_LOG}"
echo "Model: ${MODEL}" | tee -a "${MASTER_LOG}"
echo "GPU: ${GPU_ID}" | tee -a "${MASTER_LOG}"
echo "============================================================" | tee -a "${MASTER_LOG}"

# --- 1. VCD (original, fixed alpha=1.0) ---
echo "" | tee -a "${MASTER_LOG}"
echo "------------------------------------------------------------" | tee -a "${MASTER_LOG}"
echo "Method: VCD (original) | alpha=1.0 | noise_step=500" | tee -a "${MASTER_LOG}"
echo "------------------------------------------------------------" | tee -a "${MASTER_LOG}"

CMD_VCD=(
    python3 "${PROJECT_ROOT}/run_entropygate.py"
    --method vcd
    --model_name "${MODEL}"
    --seed "${SEED}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --experiment_name "vcd_chair_${TIMESTAMP}"
    --do_sample
    --load_in_4bit
    --vcd_alpha 1.0
    --vcd_beta 0.1
    --vcd_noise_step 500
    --run_chair_benchmark
    --coco_path "${COCO_PATH}"
    --coco_file "${COCO_FILE}"
    --coco_base_image_path "${COCO_BASE_IMAGE_PATH}"
    --chair_test_size "${CHAIR_TEST_SIZE}"
)
echo "CMD: ${CMD_VCD[*]}" | tee -a "${MASTER_LOG}"
"${CMD_VCD[@]}" 2>&1 | tee -a "${MASTER_LOG}"
echo "VCD done." | tee -a "${MASTER_LOG}"

# --- 2. VCD + EntropyGate (alpha in [0.5, 1.5]) ---
echo "" | tee -a "${MASTER_LOG}"
echo "------------------------------------------------------------" | tee -a "${MASTER_LOG}"
echo "Method: VCD+EntropyGate | alpha=[0.5,1.5] | eta=0.10 | tau=0.05" | tee -a "${MASTER_LOG}"
echo "------------------------------------------------------------" | tee -a "${MASTER_LOG}"

CMD_VCD_EG=(
    python3 "${PROJECT_ROOT}/run_entropygate.py"
    --method vcd_eg
    --model_name "${MODEL}"
    --seed "${SEED}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --experiment_name "vcd_eg_chair_${TIMESTAMP}"
    --do_sample
    --load_in_4bit
    --vcd_alpha 1.0
    --vcd_beta 0.1
    --vcd_noise_step 500
    --vcd_eg_alpha_min 0.5
    --vcd_eg_alpha_max 1.5
    --eta_vis 0.10
    --tau_gate 0.05
    --run_chair_benchmark
    --coco_path "${COCO_PATH}"
    --coco_file "${COCO_FILE}"
    --coco_base_image_path "${COCO_BASE_IMAGE_PATH}"
    --chair_test_size "${CHAIR_TEST_SIZE}"
)
echo "CMD: ${CMD_VCD_EG[*]}" | tee -a "${MASTER_LOG}"
"${CMD_VCD_EG[@]}" 2>&1 | tee -a "${MASTER_LOG}"
echo "VCD+EntropyGate done." | tee -a "${MASTER_LOG}"

echo "" | tee -a "${MASTER_LOG}"
echo "============================================================" | tee -a "${MASTER_LOG}"
echo "All done. Log: ${MASTER_LOG}" | tee -a "${MASTER_LOG}"
echo "============================================================" | tee -a "${MASTER_LOG}"

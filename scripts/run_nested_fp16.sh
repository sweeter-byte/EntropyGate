#!/bin/bash
set -euo pipefail

# =============================================================================
# Scheme E (nested) best config (E5) on CHAIR fp16
# Compares: vanilla / CRoPS / EntropyGate-nested (alpha_base_vis=1.5)
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
MASTER_LOG="${LOG_DIR}/nested_fp16_${TIMESTAMP}.log"

echo "============================================================" | tee -a "${MASTER_LOG}"
echo "Nested E5 fp16 CHAIR — $(date)" | tee -a "${MASTER_LOG}"
echo "Model: ${MODEL}" | tee -a "${MASTER_LOG}"
echo "Quantization: none (fp16)" | tee -a "${MASTER_LOG}"
echo "GPU: ${GPU_ID}" | tee -a "${MASTER_LOG}"
echo "============================================================" | tee -a "${MASTER_LOG}"

# --- 1. Vanilla ---
echo "" | tee -a "${MASTER_LOG}"
echo "------------------------------------------------------------" | tee -a "${MASTER_LOG}"
echo "Method: vanilla | CHAIR | fp16" | tee -a "${MASTER_LOG}"
echo "------------------------------------------------------------" | tee -a "${MASTER_LOG}"

CMD_VANILLA=(
    python3 "${PROJECT_ROOT}/run_entropygate.py"
    --method vanilla
    --model_name "${MODEL}"
    --seed "${SEED}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --experiment_name "vanilla_chair_fp16_${TIMESTAMP}"
    --do_sample
    --run_chair_benchmark
    --coco_path "${COCO_PATH}"
    --coco_file "${COCO_FILE}"
    --coco_base_image_path "${COCO_BASE_IMAGE_PATH}"
    --chair_test_size "${CHAIR_TEST_SIZE}"
)
echo "CMD: ${CMD_VANILLA[*]}" | tee -a "${MASTER_LOG}"
"${CMD_VANILLA[@]}" 2>&1 | tee -a "${MASTER_LOG}"
echo "vanilla done." | tee -a "${MASTER_LOG}"

# --- 2. CRoPS ---
echo "" | tee -a "${MASTER_LOG}"
echo "------------------------------------------------------------" | tee -a "${MASTER_LOG}"
echo "Method: crops | CHAIR | fp16" | tee -a "${MASTER_LOG}"
echo "------------------------------------------------------------" | tee -a "${MASTER_LOG}"

CMD_CROPS=(
    python3 "${PROJECT_ROOT}/run_entropygate.py"
    --method crops
    --model_name "${MODEL}"
    --seed "${SEED}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --experiment_name "crops_chair_fp16_${TIMESTAMP}"
    --do_sample
    --lambda_lang_prior 0.01
    --alpha_stat_bias 1
    --beta_cutoff 0.1
    --max_threshold_plausibility_constraint 0.95
    --run_chair_benchmark
    --coco_path "${COCO_PATH}"
    --coco_file "${COCO_FILE}"
    --coco_base_image_path "${COCO_BASE_IMAGE_PATH}"
    --chair_test_size "${CHAIR_TEST_SIZE}"
)
echo "CMD: ${CMD_CROPS[*]}" | tee -a "${MASTER_LOG}"
"${CMD_CROPS[@]}" 2>&1 | tee -a "${MASTER_LOG}"
echo "crops done." | tee -a "${MASTER_LOG}"

# --- 3. EntropyGate nested (E5 best config) ---
echo "" | tee -a "${MASTER_LOG}"
echo "------------------------------------------------------------" | tee -a "${MASTER_LOG}"
echo "Method: entropygate nested E5 | CHAIR | fp16" | tee -a "${MASTER_LOG}"
echo "------------------------------------------------------------" | tee -a "${MASTER_LOG}"

CMD_EG=(
    python3 "${PROJECT_ROOT}/run_entropygate.py"
    --method entropygate
    --model_name "${MODEL}"
    --seed "${SEED}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --experiment_name "nested_e5_chair_fp16_${TIMESTAMP}"
    --eg_scheme nested
    --alpha_base_vis 1.5
    --alpha_min_vis 0.5
    --alpha_base_txt 1.0
    --alpha_min_txt 0.0
    --eta_vis 0.10
    --eta_txt 0.40
    --tau_gate 0.05
    --gamma_decay 0.01
    --beta_cutoff_fixed 0.1
    --theta_safe 0.99
    --run_chair_benchmark
    --coco_path "${COCO_PATH}"
    --coco_file "${COCO_FILE}"
    --coco_base_image_path "${COCO_BASE_IMAGE_PATH}"
    --chair_test_size "${CHAIR_TEST_SIZE}"
)
echo "CMD: ${CMD_EG[*]}" | tee -a "${MASTER_LOG}"
"${CMD_EG[@]}" 2>&1 | tee -a "${MASTER_LOG}"
echo "entropygate nested done." | tee -a "${MASTER_LOG}"

echo "" | tee -a "${MASTER_LOG}"
echo "============================================================" | tee -a "${MASTER_LOG}"
echo "All done. Log: ${MASTER_LOG}" | tee -a "${MASTER_LOG}"
echo "============================================================" | tee -a "${MASTER_LOG}"

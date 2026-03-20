#!/bin/bash
set -euo pipefail

# =============================================================================
# EntropyGate: Full evaluation across all benchmarks and models
# Hardware: NVIDIA L40S
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---- Configurable variables ----
GPU_ID="2"
SEED="${SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
QUANTIZATION="8bit"   # none | 4bit | 8bit

# COCO dataset paths (for CHAIR benchmark)
COCO_PATH="/data1/ranmaoyin/coco2014/annotations"
COCO_FILE="${COCO_FILE:-instances_val2014.json}"
COCO_BASE_IMAGE_PATH="/data1/ranmaoyin/coco2014/val2014"
CHAIR_TEST_SIZE="${CHAIR_TEST_SIZE:-500}"

LOG_DIR="${PROJECT_ROOT}/logs"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_ROOT}"
# export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# ---- Models ----
MODELS=(
    # "llava-hf/llava-1.5-7b-hf"
    /data1/ranmaoyin/models/llava-1.5-7b-hf
    # "llava-hf/llava-1.5-13b-hf"
    # "llava-hf/llama3-llava-next-8b-hf"
    # "Qwen/Qwen2-VL-7B-Instruct"
)

# ---- Benchmarks ----
# Uncomment the benchmarks you want to run
BENCHMARKS=(
    "chair"
    # "mme"
    # "mathvista"
    # "mmmu"
)

# ---- EntropyGate default hyperparameters ----
ALPHA_BASE_VIS="${ALPHA_BASE_VIS:-1.0}"
ALPHA_BASE_TXT="${ALPHA_BASE_TXT:-1.0}"
ETA_VIS="${ETA_VIS:-0.3}"
ETA_TXT="${ETA_TXT:-0.4}"
TAU_GATE="${TAU_GATE:-0.05}"
GAMMA_DECAY="${GAMMA_DECAY:-0.01}"
BETA_BASE="${BETA_BASE:-0.05}"
BETA_RANGE="${BETA_RANGE:-0.15}"
THETA_SAFE="${THETA_SAFE:-0.99}"

# =============================================================================
# Helper: build and run a single experiment
# =============================================================================
build_cmd() {
    local model=$1
    local benchmark=$2
    local experiment_name=$3

    local CMD=(
        python3 "${PROJECT_ROOT}/run_entropygate.py"
        --model_name "${model}"
        --seed "${SEED}"
        --max_new_tokens "${MAX_NEW_TOKENS}"
        --experiment_name "${experiment_name}"
        --alpha_base_vis "${ALPHA_BASE_VIS}"
        --alpha_base_txt "${ALPHA_BASE_TXT}"
        --eta_vis "${ETA_VIS}"
        --eta_txt "${ETA_TXT}"
        --tau_gate "${TAU_GATE}"
        --gamma_decay "${GAMMA_DECAY}"
        --beta_base "${BETA_BASE}"
        --beta_range "${BETA_RANGE}"
        --theta_safe "${THETA_SAFE}"
    )

    case "${QUANTIZATION}" in
        none) ;;
        4bit) CMD+=(--load_in_4bit) ;;
        8bit) CMD+=(--load_in_8bit) ;;
        *) echo "Unsupported QUANTIZATION: ${QUANTIZATION}"; exit 1 ;;
    esac

    case "${benchmark}" in
        chair)
            CMD+=(
                --run_chair_benchmark
                --coco_path "${COCO_PATH}"
                --coco_file "${COCO_FILE}"
                --coco_base_image_path "${COCO_BASE_IMAGE_PATH}"
                --chair_test_size "${CHAIR_TEST_SIZE}"
            )
            ;;
        mme)
            CMD+=(--run_mme_benchmark)
            ;;
        mathvista)
            CMD+=(--run_mathvista_benchmark)
            ;;
        mmmu)
            CMD+=(--run_mmmu_benchmark)
            ;;
        *)
            echo "Unknown benchmark: ${benchmark}"; exit 1
            ;;
    esac

    echo "${CMD[@]}"
}

# =============================================================================
# Main loop
# =============================================================================
LOG_FILE="${LOG_DIR}/eval_${TIMESTAMP}.log"

echo "============================================================" | tee -a "${LOG_FILE}"
echo "EntropyGate Evaluation — $(date)" | tee -a "${LOG_FILE}"
echo "GPU: ${GPU_ID}  |  Quantization: ${QUANTIZATION}" | tee -a "${LOG_FILE}"
echo "Hyperparams: eta_vis=${ETA_VIS} eta_txt=${ETA_TXT} tau=${TAU_GATE}" | tee -a "${LOG_FILE}"
echo "             alpha_vis=${ALPHA_BASE_VIS} alpha_txt=${ALPHA_BASE_TXT}" | tee -a "${LOG_FILE}"
echo "             gamma=${GAMMA_DECAY} beta_base=${BETA_BASE} beta_range=${BETA_RANGE}" | tee -a "${LOG_FILE}"
echo "             theta_safe=${THETA_SAFE}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT="${MODEL##*/}"
    for BENCH in "${BENCHMARKS[@]}"; do
        EXPERIMENT_NAME="eg_${BENCH}_${TIMESTAMP}"

        echo "" | tee -a "${LOG_FILE}"
        echo "------------------------------------------------------------" | tee -a "${LOG_FILE}"
        echo "[${MODEL_SHORT}] Benchmark: ${BENCH}" | tee -a "${LOG_FILE}"
        echo "------------------------------------------------------------" | tee -a "${LOG_FILE}"

        CMD=$(build_cmd "${MODEL}" "${BENCH}" "${EXPERIMENT_NAME}")
        echo "CMD: ${CMD}" | tee -a "${LOG_FILE}"
        echo "" | tee -a "${LOG_FILE}"

        eval "${CMD}" 2>&1 | tee -a "${LOG_FILE}"

        echo "[${MODEL_SHORT}] ${BENCH} done." | tee -a "${LOG_FILE}"
    done
done

echo "" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "All evaluations complete. Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

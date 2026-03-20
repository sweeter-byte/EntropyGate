#!/bin/bash
set -euo pipefail

# =============================================================================
# CHAIR protocol supplement runs on GPU 0
# Covers:
#   - crops / greedy / seed 42
#   - vanilla / greedy / seed 42
#   - vanilla / sampling / seed 43
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID="1"
RUN_TAG="${RUN_TAG:-chair_protocol_$(date +%Y%m%d_%H%M%S)}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MODEL="${MODEL:-/data1/ranmaoyin/models/llava-1.5-7b-hf}"

COCO_PATH="${COCO_PATH:-/data1/ranmaoyin/dataset/coco2014/annotations}"
COCO_FILE="${COCO_FILE:-instances_val2014.json}"
COCO_BASE_IMAGE_PATH="${COCO_BASE_IMAGE_PATH:-/data1/ranmaoyin/dataset/coco2014/val2014}"
CHAIR_TEST_SIZE="${CHAIR_TEST_SIZE:-500}"

LOG_DIR="${PROJECT_ROOT}/logs"
MODEL_SLUG="${MODEL//\//--}"
EXPERIMENT_ROOT="${PROJECT_ROOT}/experiments/${MODEL_SLUG}/EntropyGate"

RUNS=(
    "crops greedy 42"
    "vanilla greedy 42"
    "vanilla sampling 43"
)

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_ROOT}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

mkdir -p "${LOG_DIR}" "${EXPERIMENT_ROOT}"
LOG_FILE="${LOG_DIR}/chair_protocol_gpu0_${RUN_TAG}.log"

run_experiment() {
    local method="$1"
    local decode="$2"
    local seed="$3"
    local experiment_name="${RUN_TAG}_${method}_chair_${decode}_s${seed}"
    local experiment_dir="${EXPERIMENT_ROOT}/${experiment_name}"

    if [[ -e "${experiment_dir}" ]]; then
        if [[ "${SKIP_EXISTING}" == "1" ]]; then
            echo "Skipping existing experiment: ${experiment_name}" | tee -a "${LOG_FILE}"
            return
        fi
        echo "Experiment directory already exists: ${experiment_dir}" | tee -a "${LOG_FILE}"
        echo "Set SKIP_EXISTING=1 to skip completed runs." | tee -a "${LOG_FILE}"
        exit 1
    fi

    local cmd=(
        python3 "${PROJECT_ROOT}/run_entropygate.py"
        --method "${method}"
        --model_name "${MODEL}"
        --seed "${seed}"
        --max_new_tokens "${MAX_NEW_TOKENS}"
        --experiment_name "${experiment_name}"
        --load_in_4bit
        --run_chair_benchmark
        --coco_path "${COCO_PATH}"
        --coco_file "${COCO_FILE}"
        --coco_base_image_path "${COCO_BASE_IMAGE_PATH}"
        --chair_test_size "${CHAIR_TEST_SIZE}"
    )

    if [[ "${decode}" == "sampling" ]]; then
        cmd+=(--do_sample)
    fi

    case "${method}" in
        crops)
            cmd+=(
                --lambda_lang_prior 0.01
                --alpha_stat_bias 1
                --beta_cutoff 0.1
                --max_threshold_plausibility_constraint 0.95
            )
            ;;
        vanilla|entropygate)
            ;;
        *)
            echo "Unsupported method: ${method}" | tee -a "${LOG_FILE}"
            exit 1
            ;;
    esac

    echo "" | tee -a "${LOG_FILE}"
    echo "------------------------------------------------------------" | tee -a "${LOG_FILE}"
    echo "GPU ${GPU_ID} | method=${method} | decode=${decode} | seed=${seed}" | tee -a "${LOG_FILE}"
    echo "Experiment: ${experiment_name}" | tee -a "${LOG_FILE}"
    echo "CMD: ${cmd[*]}" | tee -a "${LOG_FILE}"
    echo "------------------------------------------------------------" | tee -a "${LOG_FILE}"
    "${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
}

echo "============================================================" | tee -a "${LOG_FILE}"
echo "CHAIR Protocol Supplement — $(date)" | tee -a "${LOG_FILE}"
echo "GPU: ${GPU_ID}" | tee -a "${LOG_FILE}"
echo "Model: ${MODEL}" | tee -a "${LOG_FILE}"
echo "Run tag: ${RUN_TAG}" | tee -a "${LOG_FILE}"
echo "Experiment root: ${EXPERIMENT_ROOT}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

for run in "${RUNS[@]}"; do
    read -r method decode seed <<< "${run}"
    run_experiment "${method}" "${decode}" "${seed}"
done

echo "" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "GPU ${GPU_ID} supplement runs complete. Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

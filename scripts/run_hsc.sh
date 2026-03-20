#!/bin/bash
set -euo pipefail

# =============================================================================
# HSC: Hidden State Contrastive
#
# Stat-bias contrast in hidden-state space (pre-norm), then norm + lm_head
# projects back to logits.  Lang-prior contrast in logit space (nested,
# time decay).  RMSNorm after contrast introduces non-linearity, making
# this different from logit-space contrast.
#
# Sweeps hsc_alpha_base and hsc_alpha_min to find the optimal range.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID="${GPU_ID:-0}"
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
LOG_FILE="${LOG_DIR}/hsc_${TIMESTAMP}.log"

# Configs: (hsc_alpha_base, hsc_alpha_min, hsc_eta, hsc_tau)
CONFIGS=(
    # H1: moderate range [0.3, 1.0]
    "1.0 0.3 0.10 0.05"
    # H2: wide range [0.3, 1.5] — aggressive at high entropy
    "1.5 0.3 0.10 0.05"
    # H3: narrow high [0.5, 1.0]
    "1.0 0.5 0.10 0.05"
    # H4: very wide [0.5, 2.0]
    "2.0 0.5 0.10 0.05"
    # H5: match E5 range [0.5, 1.5]
    "1.5 0.5 0.10 0.05"
)

echo "============================================================" | tee -a "${LOG_FILE}"
echo "HSC: Hidden State Contrastive — $(date)" | tee -a "${LOG_FILE}"
echo "Model: ${MODEL}" | tee -a "${LOG_FILE}"
echo "GPU: ${GPU_ID}  |  Quantization: ${QUANTIZATION}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

IDX=0
for CFG in "${CONFIGS[@]}"; do
    IDX=$((IDX + 1))
    read -r ABASE AMIN ETA TAU <<< "${CFG}"
    EXPERIMENT_NAME="hsc_h${IDX}_${TIMESTAMP}"

    echo "" | tee -a "${LOG_FILE}"
    echo "------------------------------------------------------------" | tee -a "${LOG_FILE}"
    echo "H${IDX}: hsc_alpha_base=${ABASE} hsc_alpha_min=${AMIN} hsc_eta=${ETA} hsc_tau=${TAU}" | tee -a "${LOG_FILE}"
    echo "------------------------------------------------------------" | tee -a "${LOG_FILE}"

    CMD=(
        python3 "${PROJECT_ROOT}/run_entropygate.py"
        --method latent
        --latent_method hsc
        --model_name "${MODEL}"
        --seed "${SEED}"
        --max_new_tokens "${MAX_NEW_TOKENS}"
        --experiment_name "${EXPERIMENT_NAME}"
        --hsc_alpha_base "${ABASE}"
        --hsc_alpha_min "${AMIN}"
        --hsc_eta "${ETA}"
        --hsc_tau "${TAU}"
        --gamma_decay 0.01
        --beta_cutoff_fixed 0.1
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

    echo "H${IDX} done." | tee -a "${LOG_FILE}"
done

echo "" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "HSC experiments complete. Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

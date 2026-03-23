#!/bin/bash
set -euo pipefail

# =============================================================================
# LLH: Latent-Logit Hybrid
#
# Two-stage decoding:
#   Stage 1: Stat-bias contrast in hidden-state space (entropy-gated, like HSC)
#   Stage 2: Lang-prior contrast in logit space with ADDITIONAL entropy gate
#            on the corrected distribution (not just time decay)
#
# Double entropy gating: the hidden-space contrast is gated by original
# entropy, the logit-space lang prior is gated by corrected-distribution
# entropy.
#
# Sweeps hidden-stage alpha range.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID="5"
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
LOG_FILE="${LOG_DIR}/llh_${TIMESTAMP}.log"

# Configs: (llh_hidden_alpha_base, llh_hidden_alpha_min, llh_hidden_eta, llh_hidden_tau)
# LLH has double entropy gating: stage 1 (hidden) uses H_t, stage 2 (logit)
# uses H_corrected.  LASER's insight: higher η means the gate only fires for
# truly confused tokens, preserving model confidence on easy tokens.
CONFIGS=(
    # J1: moderate [0.3, 1.0], low η
    "1.0 0.3 0.10 0.05"
    # J2: wide [0.3, 1.5], low η
    "1.5 0.3 0.10 0.05"
    # J3: match E5 range [0.5, 1.5], low η
    "1.5 0.5 0.10 0.05"
    # J4: wide [0.3, 1.5], LASER-inspired η=0.30
    "1.5 0.3 0.30 0.10"
    # J5: moderate [0.3, 1.0], LASER-inspired η=0.50 (high threshold)
    "1.0 0.3 0.50 0.15"
)

echo "============================================================" | tee -a "${LOG_FILE}"
echo "LLH: Latent-Logit Hybrid — $(date)" | tee -a "${LOG_FILE}"
echo "Model: ${MODEL}" | tee -a "${LOG_FILE}"
echo "GPU: ${GPU_ID}  |  Quantization: ${QUANTIZATION}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

IDX=0
for CFG in "${CONFIGS[@]}"; do
    IDX=$((IDX + 1))
    read -r ABASE AMIN ETA TAU <<< "${CFG}"
    EXPERIMENT_NAME="llh_j${IDX}_${TIMESTAMP}"

    echo "" | tee -a "${LOG_FILE}"
    echo "------------------------------------------------------------" | tee -a "${LOG_FILE}"
    echo "J${IDX}: llh_alpha_base=${ABASE} llh_alpha_min=${AMIN} llh_eta=${ETA} llh_tau=${TAU}" | tee -a "${LOG_FILE}"
    echo "------------------------------------------------------------" | tee -a "${LOG_FILE}"

    CMD=(
        python3 "${PROJECT_ROOT}/run_entropygate.py"
        --method latent
        --latent_method llh
        --model_name "${MODEL}"
        --seed "${SEED}"
        --max_new_tokens "${MAX_NEW_TOKENS}"
        --experiment_name "${EXPERIMENT_NAME}"
        --llh_hidden_alpha_base "${ABASE}"
        --llh_hidden_alpha_min "${AMIN}"
        --llh_hidden_eta "${ETA}"
        --llh_hidden_tau "${TAU}"
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

    echo "J${IDX} done." | tee -a "${LOG_FILE}"
done

echo "" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "LLH experiments complete. Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

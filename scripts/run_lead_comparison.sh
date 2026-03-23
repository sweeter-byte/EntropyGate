#!/bin/bash
set -euo pipefail

# =============================================================================
# EntropyGate E5 vs LEAD Baseline Comparison
#
# Runs EntropyGate (E5), VCD, CRoPS, and Vanilla on POPE benchmark
# to produce results comparable with LEAD paper (Table 2).
#
# LEAD paper reports POPE-Random/Popular/Adversarial on:
#   - Qwen2.5-VL-7B-Instruct (not directly comparable — different arch)
#   - We run on LLaVA-1.5-7B (our primary model)
#
# Metrics: Accuracy, Precision, Recall, F1, Yes-ratio
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
MODEL="${MODEL:-/data1/ranmaoyin/models/llava-1.5-7b-hf}"

# To run with Qwen2.5-VL series models (LEAD paper comparison):
#   MODEL=/data1/ranmaoyin/models/Qwen2.5-VL-7B-Instruct bash scripts/run_lead_comparison.sh
#   MODEL=/data1/ranmaoyin/models/R1-Onevision-7B bash scripts/run_lead_comparison.sh
#   MODEL=/data1/ranmaoyin/models/Vision-R1-7B bash scripts/run_lead_comparison.sh
# Note: Requires qwen-vl-utils>=0.0.8 (pip install qwen-vl-utils)

# POPE data paths
POPE_PATH="${POPE_PATH:-/data1/ranmaoyin/dataset/pope}"
POPE_COCO_IMAGE_DIR="${POPE_COCO_IMAGE_DIR:-/data1/ranmaoyin/dataset/coco2014/val2014}"

# COCO paths (for CHAIR benchmark, optional)
COCO_PATH="${COCO_PATH:-/data1/ranmaoyin/dataset/coco2014/annotations}"
COCO_FILE="${COCO_FILE:-instances_val2014.json}"
COCO_BASE_IMAGE_PATH="${COCO_BASE_IMAGE_PATH:-/data1/ranmaoyin/dataset/coco2014/val2014}"
CHAIR_TEST_SIZE="${CHAIR_TEST_SIZE:-500}"

# Run both POPE and CHAIR by default
RUN_POPE="${RUN_POPE:-1}"
RUN_CHAIR="${RUN_CHAIR:-1}"

LOG_DIR="${PROJECT_ROOT}/logs"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_ROOT}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${LOG_DIR}/lead_comparison_${TIMESTAMP}.log"

# ---- Helper function ----
run_method() {
    local method="$1"
    local tag="$2"
    shift 2
    local extra_args=("$@")

    local experiment_name="lead_cmp_${tag}_${TIMESTAMP}"

    local cmd=(
        python3 "${PROJECT_ROOT}/run_entropygate.py"
        --method "${method}"
        --model_name "${MODEL}"
        --seed "${SEED}"
        --max_new_tokens "${MAX_NEW_TOKENS}"
        --experiment_name "${experiment_name}"
        --load_in_4bit
    )

    if [[ "${RUN_POPE}" == "1" ]]; then
        cmd+=(
            --run_pope_benchmark
            --pope_path "${POPE_PATH}"
            --pope_coco_image_dir "${POPE_COCO_IMAGE_DIR}"
            --pope_splits random popular adversarial
        )
    fi

    if [[ "${RUN_CHAIR}" == "1" ]]; then
        cmd+=(
            --run_chair_benchmark
            --coco_path "${COCO_PATH}"
            --coco_file "${COCO_FILE}"
            --coco_base_image_path "${COCO_BASE_IMAGE_PATH}"
            --chair_test_size "${CHAIR_TEST_SIZE}"
        )
    fi

    cmd+=("${extra_args[@]}")

    echo "" | tee -a "${MASTER_LOG}"
    echo "============================================================" | tee -a "${MASTER_LOG}"
    echo "Method: ${tag} (${method})" | tee -a "${MASTER_LOG}"
    echo "Experiment: ${experiment_name}" | tee -a "${MASTER_LOG}"
    echo "CMD: ${cmd[*]}" | tee -a "${MASTER_LOG}"
    echo "============================================================" | tee -a "${MASTER_LOG}"
    "${cmd[@]}" 2>&1 | tee -a "${MASTER_LOG}"
}

echo "============================================================" | tee -a "${MASTER_LOG}"
echo "LEAD Comparison Experiments — $(date)" | tee -a "${MASTER_LOG}"
echo "Model: ${MODEL}" | tee -a "${MASTER_LOG}"
echo "GPU: ${GPU_ID}" | tee -a "${MASTER_LOG}"
echo "============================================================" | tee -a "${MASTER_LOG}"

# --- 1. Vanilla (greedy) ---
run_method vanilla "vanilla"

# --- 2. VCD (LEAD also compares with VCD) ---
run_method vcd "vcd" \
    --do_sample \
    --vcd_alpha 1.0 \
    --vcd_beta 0.1 \
    --vcd_noise_step 500

# --- 3. CRoPS ---
run_method crops "crops" \
    --lambda_lang_prior 0.01 \
    --alpha_stat_bias 1 \
    --beta_cutoff 0.1 \
    --max_threshold_plausibility_constraint 0.95

# --- 4. EntropyGate E5 (our method) ---
run_method entropygate "entropygate_e5" \
    --eg_scheme nested \
    --alpha_base_vis 1.5 \
    --alpha_base_txt 1.0 \
    --alpha_min_vis 0.3 \
    --alpha_min_txt 0.1 \
    --eta_vis 0.3 \
    --eta_txt 0.4 \
    --tau_gate 0.05 \
    --gamma_decay 0.01 \
    --beta_cutoff_fixed 0.1 \
    --theta_safe 0.99

echo "" | tee -a "${MASTER_LOG}"
echo "============================================================" | tee -a "${MASTER_LOG}"
echo "All LEAD comparison experiments complete." | tee -a "${MASTER_LOG}"
echo "Log: ${MASTER_LOG}" | tee -a "${MASTER_LOG}"
echo "============================================================" | tee -a "${MASTER_LOG}"

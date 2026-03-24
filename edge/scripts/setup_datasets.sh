#!/bin/bash
set -euo pipefail

# =============================================================================
# Dataset Setup for EDGE — DMAS-aligned Experiments
#
# Required datasets:
#   1. COCO val2014 (images + annotations) — for CHAIR and POPE
#   2. POPE JSONL files (random/popular/adversarial)
#   3. MME — loaded from HuggingFace (darkyarding/MME), no local download needed
#   4. AMBER — query JSON + images + official evaluation repo
#
# Required models:
#   1. LLaVA-1.5-7B — llava-hf/llava-1.5-7b-hf
#   2. QwenVL-7B    — Qwen/Qwen2.5-VL-7B-Instruct
# =============================================================================

DATA_ROOT="${DATA_ROOT:-/data1/ranmaoyin/dataset}"
MODEL_ROOT="${MODEL_ROOT:-/data1/ranmaoyin/models}"

echo "=========================================="
echo "EDGE Dataset & Model Setup"
echo "DATA_ROOT: ${DATA_ROOT}"
echo "MODEL_ROOT: ${MODEL_ROOT}"
echo "=========================================="

# ---- 1. COCO val2014 ----
COCO_DIR="${DATA_ROOT}/coco2014"
if [[ ! -d "${COCO_DIR}/val2014" ]]; then
    echo ""
    echo "[1/4] Downloading COCO val2014 images..."
    mkdir -p "${COCO_DIR}"
    cd "${COCO_DIR}"
    wget -q http://images.cocodataset.org/zips/val2014.zip
    unzip -q val2014.zip
    rm val2014.zip
    echo "COCO val2014 images: ${COCO_DIR}/val2014"
else
    echo "[1/4] COCO val2014 images already exist: ${COCO_DIR}/val2014"
fi

if [[ ! -d "${COCO_DIR}/annotations" ]]; then
    echo "Downloading COCO annotations..."
    cd "${COCO_DIR}"
    wget -q http://images.cocodataset.org/annotations/annotations_trainval2014.zip
    unzip -q annotations_trainval2014.zip
    rm annotations_trainval2014.zip
    echo "COCO annotations: ${COCO_DIR}/annotations"
else
    echo "COCO annotations already exist: ${COCO_DIR}/annotations"
fi

# ---- 2. POPE ----
POPE_DIR="${DATA_ROOT}/pope"
if [[ ! -d "${POPE_DIR}" ]]; then
    echo ""
    echo "[2/4] Downloading POPE benchmark files..."
    mkdir -p "${POPE_DIR}"
    cd "${POPE_DIR}"
    # POPE uses COCO val2014 images (already downloaded above)
    # Download POPE question files from the official repo
    for split in random popular adversarial; do
        wget -q "https://raw.githubusercontent.com/AoiDragon/POPE/main/output/coco/coco_pope_${split}.json" \
            -O "coco_pope_${split}.json" || echo "Warning: failed to download POPE ${split}"
    done
    echo "POPE files: ${POPE_DIR}"
else
    echo "[2/4] POPE files already exist: ${POPE_DIR}"
fi

# ---- 3. MME ----
echo ""
echo "[3/4] MME: loaded from HuggingFace hub (darkyarding/MME) at runtime. No download needed."

# ---- 4. AMBER ----
AMBER_DIR="${DATA_ROOT}/amber"
if [[ ! -d "${AMBER_DIR}" ]]; then
    echo ""
    echo "[4/4] AMBER: requires manual setup. Please:"
    echo "  1. Clone: git clone https://github.com/junyangwang0410/AMBER ${AMBER_DIR}/official_repo"
    echo "  2. Download AMBER images to: ${AMBER_DIR}/images"
    echo "  3. Download query file to: ${AMBER_DIR}/data/query/query_generative.json"
else
    echo "[4/4] AMBER directory exists: ${AMBER_DIR}"
fi

# ---- Models ----
echo ""
echo "=========================================="
echo "Model Setup"
echo "=========================================="

# LLaVA-1.5-7B
LLAVA_DIR="${MODEL_ROOT}/llava-1.5-7b-hf"
if [[ ! -d "${LLAVA_DIR}" ]]; then
    echo ""
    echo "Downloading LLaVA-1.5-7B..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('llava-hf/llava-1.5-7b-hf', local_dir='${LLAVA_DIR}')
" || echo "Warning: LLaVA download failed. You can download manually:"
    echo "  huggingface-cli download llava-hf/llava-1.5-7b-hf --local-dir ${LLAVA_DIR}"
else
    echo "LLaVA-1.5-7B already exists: ${LLAVA_DIR}"
fi

# Qwen2.5-VL-7B
QWEN_DIR="${MODEL_ROOT}/Qwen2.5-VL-7B-Instruct"
if [[ ! -d "${QWEN_DIR}" ]]; then
    echo ""
    echo "Downloading Qwen2.5-VL-7B-Instruct..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct', local_dir='${QWEN_DIR}')
" || echo "Warning: Qwen download failed. You can download manually:"
    echo "  huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ${QWEN_DIR}"
else
    echo "Qwen2.5-VL-7B already exists: ${QWEN_DIR}"
fi

echo ""
echo "=========================================="
echo "Setup complete. Summary of paths:"
echo ""
echo "  COCO images:    ${COCO_DIR}/val2014"
echo "  COCO annots:    ${COCO_DIR}/annotations"
echo "  POPE files:     ${POPE_DIR}"
echo "  MME:            (HuggingFace hub, auto-download)"
echo "  AMBER:          ${AMBER_DIR}"
echo "  LLaVA-1.5-7B:   ${LLAVA_DIR}"
echo "  Qwen2.5-VL-7B:  ${QWEN_DIR}"
echo "=========================================="

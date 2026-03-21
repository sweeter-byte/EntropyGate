# EntropyGate vs LEAD 对比实验：环境准备指南

本文档说明如何准备数据集和模型，以便在 POPE / CHAIR 等 benchmark 上对比 EntropyGate E5 方法与 LEAD 论文的结果。

---

## 1. 模型

### 1.1 LLaVA-1.5-7B（我们的主力模型）

本仓库默认模型，已有支持。

```bash
# 方式一：从 HuggingFace 下载
huggingface-cli download llava-hf/llava-1.5-7b-hf --local-dir /data1/ranmaoyin/models/llava-1.5-7b-hf

# 方式二：从 hf-mirror 下载（国内镜像）
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download llava-hf/llava-1.5-7b-hf --local-dir /data1/ranmaoyin/models/llava-1.5-7b-hf
```

### 1.2 Qwen2.5-VL-7B-Instruct（LEAD 论文使用的模型）

> **注意：** 当前 EntropyGate 代码尚未完全支持 Qwen2.5-VL 的推理流水线。
> 如需在同一模型上直接对比 LEAD，需额外适配工作（见第 5 节）。
> 目前建议先在 **LLaVA-1.5-7B 上跑 POPE/CHAIR**，与 LEAD 论文报告的 VCD baseline 数据做横向对比。

```bash
# 仅供参考 — 后续适配时使用
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir /data1/ranmaoyin/models/Qwen2.5-VL-7B-Instruct
```

---

## 2. 数据集

### 2.1 POPE（核心对比 benchmark）

POPE 是 LEAD 论文和本仓库共同可用的幻觉评测基准。

**下载方式：**

```bash
# 从 POPE 官方仓库获取数据文件
git clone https://github.com/AoiDragon/POPE.git /tmp/pope_repo

# 将 POPE 数据文件复制到数据目录
mkdir -p /data1/ranmaoyin/dataset/pope
cp /tmp/pope_repo/output/coco/coco_pope_random.json /data1/ranmaoyin/dataset/pope/
cp /tmp/pope_repo/output/coco/coco_pope_popular.json /data1/ranmaoyin/dataset/pope/
cp /tmp/pope_repo/output/coco/coco_pope_adversarial.json /data1/ranmaoyin/dataset/pope/
```

**数据格式（JSONL）：**
```json
{"question_id": 1, "image": "COCO_val2014_000000453302.jpg", "text": "Is there a snowboard in the image?", "label": "yes"}
```

**目录结构：**
```
/data1/ranmaoyin/dataset/pope/
├── coco_pope_random.json        # 3000 samples (500 images × 6 questions)
├── coco_pope_popular.json       # 3000 samples
└── coco_pope_adversarial.json   # 3000 samples
```

### 2.2 COCO 2014 验证集（POPE 和 CHAIR 共用）

POPE 的问题基于 COCO val2014 图片，CHAIR 评测也需要 COCO 数据。

```bash
mkdir -p /data1/ranmaoyin/dataset/coco2014

# 下载验证集图片
wget http://images.cocodataset.org/zips/val2014.zip -O /tmp/val2014.zip
unzip /tmp/val2014.zip -d /data1/ranmaoyin/dataset/coco2014/

# 下载标注文件（CHAIR 评测需要）
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip -O /tmp/annotations.zip
unzip /tmp/annotations.zip -d /data1/ranmaoyin/dataset/coco2014/
```

**最终目录结构：**
```
/data1/ranmaoyin/dataset/coco2014/
├── val2014/                         # 40504 张图片
│   ├── COCO_val2014_000000000042.jpg
│   └── ...
└── annotations/
    ├── instances_val2014.json       # CHAIR 评测需要
    ├── captions_val2014.json
    └── ...
```

### 2.3 AMBER（可选，已有支持）

```bash
# 参考现有脚本中的路径
# 默认路径：/data1/ranmaoyin/dataset/amber/
# 需要：
#   - data/query/query_generative.json
#   - images/
#   - official_repo/ (AMBER 官方评测脚本)
```

---

## 3. 运行对比实验

### 3.1 快速运行（POPE only）

```bash
cd /home/user/EntropyGate

# 仅跑 POPE，不跑 CHAIR
RUN_CHAIR=0 GPU_ID=0 bash scripts/run_lead_comparison.sh
```

### 3.2 完整运行（POPE + CHAIR）

```bash
GPU_ID=0 bash scripts/run_lead_comparison.sh
```

### 3.3 自定义参数

```bash
# 使用其他 GPU、指定模型路径
GPU_ID=2 \
MODEL=/your/path/to/llava-1.5-7b-hf \
POPE_PATH=/your/path/to/pope \
POPE_COCO_IMAGE_DIR=/your/path/to/coco2014/val2014 \
bash scripts/run_lead_comparison.sh
```

### 3.4 单独运行某个方法

```bash
# 仅运行 EntropyGate E5 on POPE
python run_entropygate.py \
    --method entropygate \
    --model_name /data1/ranmaoyin/models/llava-1.5-7b-hf \
    --load_in_4bit \
    --eg_scheme nested \
    --alpha_base_vis 1.5 \
    --alpha_base_txt 1.0 \
    --alpha_min_vis 0.3 \
    --alpha_min_txt 0.1 \
    --eta_vis 0.3 \
    --eta_txt 0.4 \
    --tau_gate 0.05 \
    --max_new_tokens 64 \
    --experiment_name pope_eg_e5_test \
    --run_pope_benchmark \
    --pope_path /data1/ranmaoyin/dataset/pope \
    --pope_coco_image_dir /data1/ranmaoyin/dataset/coco2014/val2014
```

---

## 4. 输出与结果对比

### 4.1 结果目录

```
experiments/
└── data1--ranmaoyin--models--llava-1.5-7b-hf/
    └── EntropyGate/
        └── POPE/
            ├── lead_cmp_vanilla_*/
            │   ├── pope_generations.jsonl
            │   └── pope_results.json
            ├── lead_cmp_vcd_*/
            ├── lead_cmp_crops_*/
            └── lead_cmp_entropygate_e5_*/
```

### 4.2 POPE 指标说明

| 指标 | 说明 | LEAD 论文也报告 |
|------|------|----------------|
| Accuracy | 正确率 | ✅ |
| Precision | 精确率 | ✅ |
| Recall | 召回率 | ✅ |
| F1 | F1 分数 | ✅ |
| Yes-ratio | 回答 "yes" 的比例 | ✅ |

### 4.3 与 LEAD 论文数据对比

LEAD 论文 Table 2 报告了以下方法在 POPE 上的结果（Qwen2.5-VL-7B 模型）：

| Method | POPE-R Acc | POPE-P Acc | POPE-A Acc |
|--------|-----------|-----------|-----------|
| Greedy | (论文数据) | ... | ... |
| VCD | (论文数据) | ... | ... |
| SID | (论文数据) | ... | ... |
| LEAD | (论文数据) | ... | ... |

**注意：** 由于模型不同（LLaVA-1.5-7B vs Qwen2.5-VL-7B），绝对数值不可直接对比。
建议关注：
1. **相对提升幅度**：各方法相对于 Vanilla 的提升百分比
2. **VCD baseline 对齐**：两边都有 VCD，可作为锚点校准

---

## 5. 后续扩展（可选）

### 5.1 添加 Qwen2.5-VL 支持

如需在与 LEAD 完全相同的模型上对比，需要：

1. 在 `constants/image_token_constants.py` 中注册 Qwen2.5-VL 的 image token ID
2. 确认 `_build_full_inputs()` 对 Qwen2.5-VL 的 chat template 兼容性
3. 测试 EntropyGate 的 FastV/TextMask 在 Qwen2.5-VL attention 结构上的适配
   （已有 `methods/model_forward/crops_qwen_forward.py`，需验证版本兼容）

### 5.2 添加 MMHalBench

LEAD 论文还报告了 MMHalBench 结果，可后续添加该 benchmark。

### 5.3 添加 SID / MemVR baseline

LEAD 论文对比了 SID 和 MemVR 方法，可考虑实现作为额外 baseline。

---

## 6. 依赖检查

```bash
# 确认依赖已安装
pip install -r requirements.txt

# POPE 不需要额外依赖，使用标准 JSON 解析
# 确认 accelerate 可用（多卡推理）
python -c "from accelerate import PartialState; print('OK')"
```

---

## 7. 文件清单

本次新增/修改的文件：

| 文件 | 说明 |
|------|------|
| `benchmark/pope_benchmark.py` | POPE benchmark 数据加载、评测实现 |
| `run_entropygate.py` | 新增 POPE 相关参数和 `run_pope_benchmark()` |
| `scripts/run_lead_comparison.sh` | 一键对比实验脚本 |
| `docs/lead_comparison_setup.md` | 本文档 |

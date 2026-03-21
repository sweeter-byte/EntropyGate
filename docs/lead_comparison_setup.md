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

### 1.2 LEAD 论文使用的模型

LEAD 论文**不使用 Qwen2.5-VL**，而是在以下 **MLRM（多模态大推理模型）** 上评测：

| 模型 | 类型 | 说明 |
|------|------|------|
| **R1-Onevision-7B** | MLRM | LEAD 主要评测模型 |
| **Vision-R1-7B** | MLRM | 第二评测模型 |
| VL-Rethinker-7B | MLRM | 附加评测 |
| VL-Cogito-7B | MLRM | 附加评测 |
| OpenVLThinker-7B | MLRM | 附加评测 |

> **重要：** LEAD 针对的是推理模型（R1 系列），与我们的传统 VLM（LLaVA）是不同代际的模型。
> 因此绝对数值**不可直接对比**，应关注各方法相对于 Vanilla baseline 的**提升幅度**。
> 此外，两篇工作共同对比的 baseline 是 **VCD**，可以作为校准锚点。

```bash
# 如需下载 R1-Onevision-7B（可选，用于在相同模型上跑 EntropyGate）
huggingface-cli download Qwen/QVQ-72B-Preview --local-dir /data1/ranmaoyin/models/R1-Onevision-7B
# 注意：需先确认 HuggingFace 上的确切模型 ID
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

| 指标 | 说明 | LEAD 论文报告 |
|------|------|--------------|
| Accuracy | 正确率 | ✅（仅报告此指标） |
| Precision | 精确率 | ❌ |
| Recall | 召回率 | ❌ |
| F1 | F1 分数 | ❌ |
| Yes-ratio | 回答 "yes" 的比例 | ❌ |

> LEAD 论文在 POPE 上仅报告 Accuracy。我们额外报告 Precision/Recall/F1/Yes-ratio
> 以便更全面地分析方法特性（如是否存在 yes-bias）。

### 4.3 与 LEAD 论文数据对比

**LEAD 论文 Table 2 POPE 结果**（来源：arXiv:2603.13366）

**R1-Onevision-7B 模型：**

| Method | POPE-R ↑ | POPE-P ↑ | POPE-A ↑ |
|--------|----------|----------|----------|
| Baseline (Greedy) | 84.6 | 84.0 | 82.5 |
| + VCD | 84.4 | 83.8 | 82.3 |
| + MemVR | 82.3 | 85.0 | 83.5 |
| + SID | 85.0 | 84.7 | 81.9 |
| + **LEAD** | **85.9 (+1.3)** | **85.3 (+1.3)** | **83.9 (+1.4)** |

**Vision-R1-7B 模型：**

| Method | POPE-R ↑ | POPE-P ↑ | POPE-A ↑ |
|--------|----------|----------|----------|
| Baseline | 88.0 | 85.2 | 84.0 |
| + **LEAD** | **91.4 (+3.4)** | **88.3 (+3.1)** | **87.7 (+3.7)** |

**其他模型：**

| Model + LEAD | POPE-R ↑ | POPE-P ↑ | POPE-A ↑ |
|--------------|----------|----------|----------|
| VL-Rethinker-7B | 85.5 → 86.2 | 81.8 → 85.1 | 82.8 → 84.9 |
| VL-Cogito-7B | 85.0 → 86.3 | 85.0 → 86.6 | 84.1 → 86.1 |
| OpenVLThinker-7B | 82.4 → 84.1 | 82.5 → 83.5 | 79.1 → 80.2 |

### 4.4 对比分析建议

**模型差异：** LEAD 使用 MLRM（R1 推理模型），我们使用传统 VLM（LLaVA-1.5-7B）。
绝对数值不可直接比较，建议关注：

1. **相对提升幅度**：各方法相对 Vanilla 的 Accuracy 提升
   - LEAD 在 R1-Onevision 上: POPE-R +1.3%, POPE-P +1.3%, POPE-A +1.4%
   - EntropyGate E5 在 LLaVA 上: (待实验)
2. **VCD baseline 作为锚点**：两边都对比了 VCD，可校准提升效果
   - LEAD 论文中 VCD 在 R1 上几乎无提升甚至下降（POPE-R: 84.6→84.4）
   - 这说明 VCD 对推理模型效果有限，而 LEAD 的提升更显著
3. **LEAD 代码仓库没有开源 POPE 评测代码**，其 POPE 结果的具体实现细节未知

---

## 5. 后续扩展（可选）

### 5.1 在 R1 推理模型上运行 EntropyGate

如需在与 LEAD 完全相同的模型（R1-Onevision-7B）上对比，需要：

1. 在 `constants/image_token_constants.py` 中注册 R1-Onevision 的 image token ID
2. 确认 `_build_full_inputs()` 对 Qwen2-VL 系列 chat template 的兼容性
3. 测试 EntropyGate 的 FastV/TextMask 在 Qwen2-VL attention 结构上的适配
   （已有 `methods/model_forward/crops_qwen_forward.py`，需验证版本兼容）
4. 注意 R1 模型输出通常包含 `<think>...</think>` 推理链，需调整 POPE 回答提取逻辑

### 5.2 添加 LEAD 论文的其他对比 benchmark

LEAD Table 2 还报告了以下 benchmark，可考虑添加：

| Benchmark | 类型 | 优先级 |
|-----------|------|--------|
| **MMHalu** | 幻觉评估 (score 0-6) | 高 |
| **Bingo** | 幻觉评估 (score 1-5) | 高 |
| **MMVP** | 通用理解 | 中 |
| **RealWorldQA** | 通用理解 | 中 |
| **VMCBench** | 通用理解 | 低 |

### 5.3 添加 SID / MemVR baseline

LEAD 论文对比了 VCD、SID、MemVR 三个 baseline。
- VCD: 已实现 ✅
- SID: 待实现
- MemVR: 待实现

### 5.4 LEAD 论文 Table 3 — MathVista 对比

我们已有 MathVista benchmark 支持。LEAD 论文的 MathVista 结果：

| Model | Baseline | + LEAD |
|-------|----------|--------|
| R1-Onevision-7B | 64.1 | 66.4 (+2.3) |
| Vision-R1-7B | 73.5 | 74.9 (+1.4) |

可直接在 LLaVA 上运行 MathVista 做对比。

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

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

LEAD 论文在以下 **MLRM（多模态大推理模型）** 上评测，这些模型**全部基于 Qwen2.5-VL-7B 架构**微调而来：

| 模型 | HuggingFace ID | 基座模型 | 训练方法 | 说明 |
|------|---------------|---------|---------|------|
| **R1-Onevision-7B** | `Fancy-MLLM/R1-Onevision-7B` | Qwen2.5-VL-7B-Instruct | SFT + RL | LEAD 主要评测模型 |
| **Vision-R1-7B** | `Osilly/Vision-R1-7B` | Qwen2.5-VL-7B-Instruct | Cold-start SFT + GRPO | 第二评测模型 (ICLR 2026) |
| VL-Rethinker-7B | `TIGER-Lab/VL-Rethinker-7B` | Qwen2.5-VL-7B-Instruct | GRPO-SSR + Forced Rethinking | 附加评测 |
| VL-Cogito-7B | `csyrf/VL-Cogito` | Qwen2.5-VL-7B-Instruct | Progressive Curriculum RL | 附加评测 |
| OpenVLThinker-7B | `ydeng9/OpenVLThinker-7B` | Qwen2.5-VL-7B | Iterative SFT-RL | 附加评测 |

> **关键发现：** 所有 LEAD 模型都是 Qwen2.5-VL-7B 的推理增强微调版本（~8B 参数，BF16）。
> 它们与我们的传统 VLM（LLaVA）是不同代际的模型。
> 因此绝对数值**不可直接对比**，应关注各方法相对于 Vanilla baseline 的**提升幅度**。
> 此外，两篇工作共同对比的 baseline 是 **VCD**，可以作为校准锚点。

#### 模型下载方式

推荐优先下载 **R1-Onevision-7B**（LEAD 主力模型）和 **Vision-R1-7B**（第二评测模型）：

```bash
# ========== 方式一：从 HuggingFace 直接下载 ==========

# R1-Onevision-7B（LEAD 主力模型，约 16GB）
huggingface-cli download Fancy-MLLM/R1-Onevision-7B \
    --local-dir /data1/ranmaoyin/models/R1-Onevision-7B

# Vision-R1-7B（第二评测模型，约 16GB）
huggingface-cli download Osilly/Vision-R1-7B \
    --local-dir /data1/ranmaoyin/models/Vision-R1-7B

# ========== 方式二：国内镜像下载（推荐） ==========

# R1-Onevision-7B
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Fancy-MLLM/R1-Onevision-7B \
    --local-dir /data1/ranmaoyin/models/R1-Onevision-7B

# Vision-R1-7B
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Osilly/Vision-R1-7B \
    --local-dir /data1/ranmaoyin/models/Vision-R1-7B

# ========== 可选：其他 LEAD 评测模型 ==========

# VL-Rethinker-7B
huggingface-cli download TIGER-Lab/VL-Rethinker-7B \
    --local-dir /data1/ranmaoyin/models/VL-Rethinker-7B

# VL-Cogito-7B
huggingface-cli download csyrf/VL-Cogito \
    --local-dir /data1/ranmaoyin/models/VL-Cogito-7B

# OpenVLThinker-7B
huggingface-cli download ydeng9/OpenVLThinker-7B \
    --local-dir /data1/ranmaoyin/models/OpenVLThinker-7B

# ========== 基座模型（可选，用于 vanilla baseline 对比） ==========

# Qwen2.5-VL-7B-Instruct（所有 LEAD 模型的基座）
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
    --local-dir /data1/ranmaoyin/models/Qwen2.5-VL-7B-Instruct
```

#### 硬件需求

| 加载方式 | 显存需求 | 说明 |
|---------|---------|------|
| BF16 全精度 | ~16-18 GB | 需要 A100/A6000 等 |
| 8-bit 量化 (`--load_in_8bit`) | ~10-12 GB | RTX 3090/4090 可用 |
| 4-bit 量化 (`--load_in_4bit`) | ~6-8 GB | RTX 3080/4080 可用 |

> **推荐：** 使用 4-bit 量化可在大多数消费级 GPU 上运行。

#### 使用 LEAD 模型前的适配工作

这些模型**可以用于我们的 POPE 实验**，但由于架构差异（Qwen2.5-VL vs LLaVA），需要以下适配：

1. **已有基础：** `methods/model_forward/crops_qwen_forward.py` 已实现 Qwen2-VL 的 forward hook
2. **需注册 image token ID：** 在 `constants/image_token_constants.py` 中添加 Qwen2.5-VL 的 image token（Qwen2-VL 使用 `<|image_pad|>` token，ID 需从 tokenizer 中读取）
3. **需适配 chat template：** Qwen2.5-VL 使用 `processor.apply_chat_template()` 构建输入，与 LLaVA 的 prompt 格式不同
4. **需处理 `<think>` 标签：** 推理模型输出通常包含 `<think>...</think>` 推理链，POPE 答案提取逻辑需跳过 think 块，仅从最终回答中提取 yes/no
5. **需安装额外依赖：** `pip install qwen-vl-utils`

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

### 5.1 在 LEAD 论文的 R1 推理模型上运行 EntropyGate

**可行性：可以使用 LEAD 的模型。** 所有 LEAD 模型都基于 Qwen2.5-VL-7B，我们已有 Qwen2-VL forward hook 支持。

**具体适配步骤：**

**Step 1：下载模型**（参见 1.2 节的下载命令）

**Step 2：注册 image token ID**

Qwen2.5-VL 使用 `<|image_pad|>` 作为图像占位 token。需在 `constants/image_token_constants.py` 中添加：

```python
# 可通过以下代码查询实际 token ID：
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Fancy-MLLM/R1-Onevision-7B")
print(tokenizer.convert_tokens_to_ids("<|image_pad|>"))
# 通常为 151655
```

**Step 3：适配输入构建**

Qwen2.5-VL 使用不同的 chat template，需修改 `_build_full_inputs()` 或新增分支：

```python
# Qwen2.5-VL 的输入格式
messages = [{"role": "user", "content": [
    {"type": "image", "image": image_path},
    {"type": "text", "text": "Is there a dog in the image?"}
]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt")
```

**Step 4：处理推理模型的 `<think>` 输出**

R1 系列模型会输出 `<think>思考过程...</think>最终回答` 格式。POPE 答案提取需跳过 think 块：

```python
import re
def extract_answer_from_reasoning_model(output: str) -> str:
    """从推理模型输出中提取最终回答（跳过 <think> 块）"""
    # 移除 <think>...</think> 块
    answer = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL).strip()
    return answer
```

**Step 5：验证 forward hook 兼容性**

`methods/model_forward/crops_qwen_forward.py` 已实现 Qwen2-VL 的 attention mask 操作，
但需验证其与 Qwen2.5-VL 的兼容性（API 可能有细微变化）。

**Step 6：安装额外依赖**

```bash
pip install qwen-vl-utils>=0.0.8
```

**推荐实验顺序：**
1. 先在 LLaVA-1.5-7B 上完成所有方法的 POPE 评测（无需适配）
2. 再在 R1-Onevision-7B 上运行 vanilla + EntropyGate，与 LEAD 论文直接对比
3. 可选：在 Qwen2.5-VL-7B-Instruct（基座模型）上跑，观察推理微调的影响

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
# 确认基础依赖已安装
pip install -r requirements.txt

# POPE 不需要额外依赖，使用标准 JSON 解析
# 确认 accelerate 可用（多卡推理）
python -c "from accelerate import PartialState; print('OK')"

# 如需使用 LEAD 的 Qwen2.5-VL 系列模型，还需安装：
pip install qwen-vl-utils>=0.0.8

# 验证 Qwen2.5-VL 模型可加载
python -c "from transformers import Qwen2_5_VLForConditionalGeneration; print('Qwen2.5-VL OK')"
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

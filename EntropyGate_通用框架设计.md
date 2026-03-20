# EntropyGate 通用框架设计思路

> 日期：2026-03-19
> 目标：将 EntropyGate 从"CRoPS 的改进版"升级为"对比解码的通用自适应层"

---

## 一、核心观察

所有对比解码方法共享同一个公式骨架：

```
logits_final = logits_orig + α * (logits_orig - logits_hallucination)
```

不同方法的区别仅在于：
- 如何构造 `logits_hallucination`（去视觉 token、加噪声、取不同层、扰动注意力...）
- `α` 如何确定（固定值、时间衰减、手动调...）

它们共享同一个问题：**α 是预设的，与模型当前 token 的认知状态无关**。

---

## 二、问题分析

通过实验观察到：
- 模型在生成不同 token 时的不确定性（熵）是高度动态的
- 在高确信 token（低熵，如介词 "in"、"on"）上施加对比解码 → 引入噪声，损害生成质量
- 在低确信 token（高熵，如判断物体是否存在）上对比强度不够 → 无法有效抑制幻觉
- 固定 α 是一个 one-size-fits-all 的妥协，无法同时满足两种场景

---

## 三、方法：EntropyGate 作为通用自适应层

EntropyGate 是一个即插即用的模块，可以套在任意对比解码方法上：

```python
# 任意对比解码方法的原始公式
logits_final = logits_orig + α_fixed * contrast_signal

# 套上 EntropyGate 后
H_t = normalized_entropy(softmax(logits_orig))
α_t = α_min + (α_max - α_min) * sigmoid((H_t - η) / τ)
logits_final = logits_orig + α_t * contrast_signal
```

对于每个具体方法，EntropyGate 只替换 α 的计算方式，不改变 contrast_signal 的构造。

---

## 四、适用方法一览

| 基础方法 | 会议 | contrast_signal 构造 | 原始 α | + EntropyGate |
|---------|------|---------------------|--------|--------------|
| VCD | CVPR 2024 | 对原始图像加高斯噪声，对比 orig vs noised | 固定 (如 1.0) | α_t = f(H_t) |
| DoLa | ICLR 2024 | 对比成熟层 vs 早期层的 logits | 固定 (如 0.1) | α_t = f(H_t) |
| M3ID | 2024 | 完全移除图像 token，对比 orig vs no_image | 固定 | α_t = f(H_t) |
| SID | ICLR 2025 | 注意力选择性扰动视觉 token | 固定 | α_t = f(H_t) |
| CRoPS | 2024 | 两步嵌套：stat_bias + lang_prior 双对比 | 固定 α + 时间衰减 | α_t = f(H_t) |

---

## 五、论文叙事

### Problem
现有对比解码方法都用固定或启发式的对比强度，忽略了模型逐 token 的不确定性。
我们通过实验发现，在模型高确信的 token 上施加对比解码会引入噪声损害质量，
在模型低确信的 token 上对比强度又不够。

### Method
提出 EntropyGate，一个即插即用的自适应对比强度调制模块。
核心改动：把固定 α 替换为 entropy-gated α_t。

### Experiments
在 5 种对比解码方法 × 3+ 个模型 × 3+ 个 benchmark 上验证，
EntropyGate 一致性地带来提升。

### Contributions
1. **Analysis**：系统性地揭示"固定对比强度"的问题——逐 token 熵分析展示对比解码在哪些 token 上帮倒忙
2. **Method**：提出 EntropyGate，通用的 entropy-gated 对比强度调制机制
3. **Experiments**：多方法/多模型/多 benchmark 验证通用性，证明这是 general principle 而非某个方法的 trick

---

## 六、实验计划

### 6.1 基础方法集成（按优先级）

1. **VCD + EntropyGate**：最容易实现，VCD 只有一个 α，代码简单
2. **DoLa + EntropyGate**：单 α，且是 NLG 领域，能证明跨任务通用性
3. **CRoPS + EntropyGate**：已完成（即当前 Scheme E5）
4. **SID + EntropyGate**：锦上添花

### 6.2 模型覆盖

- LLaVA-1.5-7B
- LLaVA-1.5-13B
- Qwen2-VL（或 InternVL）

### 6.3 Benchmark 覆盖

- CHAIR (COCO val2014, 500 samples)
- AMBER (generative)
- POPE (binary VQA)
- （可选）MME, LLaVA-Bench

### 6.4 Ablation 设计

- 固定 α vs entropy-gated α（核心 ablation）
- η 和 τ 的敏感性分析
- 不同熵度量方式（归一化熵 vs 原始熵 vs top-k 熵）
- 高熵/低熵 token 上的对比解码效果可视化

---

## 七、与当前 E5 的关系

当前 E5 (nested + entropy gate on CRoPS) 是通用框架在 CRoPS 上的一个实例。
论文中 CRoPS + EntropyGate 的结果直接复用 E5 的数据，无需重跑。

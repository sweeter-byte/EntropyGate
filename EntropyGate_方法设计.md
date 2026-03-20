# EntropyGate: Uncertainty-Aware Visual Feature Gating for Hallucination Mitigation in Vision-Language Models

---

## 1. 动机 (Motivation)

### 1.1 现有方法的局限性

当前 training-free 幻觉治理方法（如 CRoPS、VCD、SID、M3ID）共享一个隐含假设：**在每一步解码中，视觉特征的参与方式是预设的、与模型当前状态无关的**。具体而言：

- **CRoPS** 使用固定的 α^(1) 对视觉缺失幻觉模型做对比，虽然 α_t^(2) 随时间递增，但这是一个基于时间步的启发式调度，并非基于模型实际的认知状态。
- **VCD/M3ID** 完全移除视觉 token 构建幻觉模型，对比强度在整个生成过程中保持不变。
- **SID** 通过注意力权重选择低重要性视觉 token，但选择策略是静态的（固定保留比例）。

这些方法忽略了一个关键事实：**模型在生成不同 token 时的不确定性是高度动态的**。当模型对下一个 token 高度确信时（如生成常见介词"in"、"on"），视觉特征的干预是多余的，甚至可能引入噪声；而当模型不确定时（如决定场景中是否存在某个物体），恰恰是视觉特征最需要发挥作用的时刻——但也是最容易产生幻觉的时刻。

### 1.2 来自 Laser 的启发

Laser 论文提出了两个与我们高度相关的核心洞察：

**洞察 A: 熵作为认知状态的度量。** Laser 的 Entropy-Regularized Intervention 机制证明，模型输出分布的归一化熵 H(Q_t) 是衡量模型"是否困惑"的有效信号。当 H(Q_t) > η 时，模型处于高不确定性状态，需要更强的外部引导；当 H(Q_t) ≤ η 时，模型已掌握上下文，可以自主推理。这一机制在 Laser 中用于决定是否注入硬目标，我们将其迁移到幻觉治理场景中，用于决定视觉特征的参与强度。

**洞察 B: 动态窗口的"先全局后局部"思想。** Laser 的 DWAL 让模型在推理早期维持全局语义的"概率叠加态"，后期才收敛到局部精确。这启发我们：对比解码中视觉特征的参与也应该是动态的——不是简单地随时间衰减（如 CRoPS 的 α_t^(2)），而是根据模型在每个 token 上的实际认知状态来自适应调节。

### 1.3 核心思想

我们提出 **EntropyGate**：在 CRoPS 的对比解码框架中引入一个 **token-level 的熵门控机制**，根据模型在每一步生成时的不确定性（通过输出分布的熵来度量），动态缩放视觉特征对当前 token 的影响强度。

直觉上：
- **模型确信时（低熵）**：视觉对比信号减弱 → 避免过度干预导致的信息损失
- **模型困惑时（高熵）**：视觉对比信号增强 → 在最需要的时刻提供更强的视觉接地

这将 CRoPS 中固定/启发式的对比强度调度，替换为一个数据驱动的、逐 token 自适应的门控策略。

---

## 2. 方法设计 (Method Design)

### 2.1 总体框架

EntropyGate 建立在 CRoPS 的广义对比解码框架之上，保留其双幻觉模型设计（视觉缺失模型 + 文本-视觉联合缺失模型），但将固定的对比系数替换为熵驱动的动态门控。

整体流程（每步解码）：

```
Step 1: 原始模型前向传播 → p_t^{orig}, 计算熵 H_t
Step 2: 视觉缺失幻觉模型前向传播 → p_t^{vis-hal}
Step 3: 文本-视觉联合缺失幻觉模型前向传播 → p_t^{vis-txt-hal}
Step 4: 熵门控计算 → g_t^{vis}, g_t^{txt}
Step 5: 门控对比解码 → p_t^{final}
```

### 2.2 熵门控机制 (Entropy Gating)

#### 2.2.1 Token-Level 不确定性度量

在每一步 t，原始模型输出 logits z_t，我们计算归一化熵：

```
p_t = softmax(z_t)
H_raw(t) = -Σ_v p_t(v) · log p_t(v)
H_t = H_raw(t) / log|V|
```

其中 |V| 是词表大小，H_t ∈ [0, 1]。H_t 接近 0 表示模型高度确信，接近 1 表示高度不确定。

#### 2.2.2 门控函数设计

我们设计两个独立的门控信号，分别控制两个幻觉模型的对比强度：

**视觉对比门控 g_t^{vis}（控制 p_t^{vis-hal} 的对比强度）：**

```
g_t^{vis} = α_base^{vis} · σ((H_t - η_vis) / τ_gate)
```

其中 σ 是 sigmoid 函数，η_vis 是视觉门控的熵阈值，τ_gate 是温度参数控制门控的锐度。当 H_t > η_vis 时，g_t^{vis} 趋近 α_base^{vis}（全力对比）；当 H_t < η_vis 时，g_t^{vis} 趋近 0（减弱对比）。

**文本-视觉联合对比门控 g_t^{txt}（控制 p_t^{vis-txt-hal} 的对比强度）：**

```
g_t^{txt} = α_base^{txt} · σ((H_t - η_txt) / τ_gate) · (1 - e^{-γt}) / e^{-γt}
```

这里保留了 CRoPS 中 α_t^(2) 的时间递增特性（因为文本依赖确实随时间增强），但额外乘以熵门控因子。这意味着即使在生成后期，如果模型对当前 token 高度确信，文本对比信号也会被抑制。

#### 2.2.3 门控对比解码公式

最终的 EntropyGate 解码公式为：

```
log p_t^{EG} = (1 + g_t^{vis} + g_t^{txt}) · log p_t^{orig}
               - g_t^{vis} · log p_t^{vis-hal}
               - g_t^{txt} · log p_t^{vis-txt-hal}
```

对比 CRoPS 的原始公式：

```
log p_t^{CRoPS} = (1 + α^(1) + α_t^(2)) · log p_t^{orig}
                  - α^(1) · log p_t^{vis-hal}
                  - α_t^(2) · log p_t^{vis-txt-hal}
```

关键区别：CRoPS 的 α^(1) 是常数，α_t^(2) 仅依赖时间步；而 EntropyGate 的 g_t^{vis} 和 g_t^{txt} 都依赖于当前 token 的实际不确定性。

### 2.3 Plausibility Constraint 的改进

CRoPS 使用一个硬阈值 plausibility constraint：当 max(p_t) > θ_plaus 时，直接跳过对比解码，使用原始 logits。这是一个 0/1 的二值决策。

EntropyGate 将其软化为连续门控的一部分：当模型高度确信（低熵）时，门控值自然趋近 0，对比信号自动减弱，无需硬阈值切换。但我们仍保留一个极端情况的安全阈值 θ_safe（设为 0.99），仅在模型几乎完全确定时才完全跳过对比。

### 2.4 Cutoff Threshold 的自适应

CRoPS 使用固定的 β_cutoff = 0.1 来截断低概率 token。我们将其与熵关联：

```
β_t = β_base + β_range · (1 - H_t)
```

当模型确信时（低熵），β_t 较大，截断更激进（保留更少候选）；当模型不确定时（高熵），β_t 较小，保留更多候选以供对比解码选择。

### 2.5 超参数汇总

| 超参数 | 含义 | 默认值 | 来源 |
|--------|------|--------|------|
| α_base^{vis} | 视觉对比基础强度 | 1.0 | 对应 CRoPS 的 α^(1) |
| α_base^{txt} | 文本对比基础强度 | 1.0 | 对应 CRoPS 的 α_t^(2) 的基础部分 |
| η_vis | 视觉门控熵阈值 | 0.3 | 借鉴 Laser 的 η=0.6，但此处在词表空间，阈值更低 |
| η_txt | 文本门控熵阈值 | 0.4 | 略高于 η_vis，文本对比更容易触发 |
| τ_gate | 门控温度 | 0.05 | 控制 sigmoid 锐度 |
| γ | 时间衰减系数 | 0.01 | 沿用 CRoPS 的 λ_lang_prior |
| β_base | 截断阈值基础值 | 0.05 | 低于 CRoPS 的 0.1 |
| β_range | 截断阈值范围 | 0.15 | β_t ∈ [0.05, 0.20] |
| θ_safe | 安全跳过阈值 | 0.99 | 高于 CRoPS 的 0.95 |

### 2.6 与 CRoPS 代码框架的集成方案

EntropyGate 复用 CRoPS 的全部代码框架，仅需修改以下文件：

**修改 1: `constants/entropygate_constants.py`（新增）**

新增熵门控相关的超参数常量定义。

**修改 2: `methods/generation_configs/contrastive_generation_config.py`**

在 `GenerationConfigContrastive` 中新增 EntropyGate 相关参数（η_vis, η_txt, τ_gate, β_base, β_range, θ_safe）。

**修改 3: `methods/samplers/entropygate_sample.py`（新增，基于 crops_sample.py）**

核心修改在解码循环中。原 CRoPS 的关键代码段：

```python
# CRoPS 原始逻辑
if probs_next_token.max(...) > max_threshold_plausibility_constraint:
    final_logits = next_token_logits
else:
    gamma_lang_prior = math.exp(-lambda_lang_prior * time_step)
    final_logits = log_probs + (1-gamma)/gamma * (log_probs - log_probs_lang_prior)
    final_logits = (1+alpha) * final_logits - alpha * log_probs_stat_bias
```

替换为 EntropyGate 逻辑：

```python
# EntropyGate 逻辑
probs = torch.softmax(next_token_logits, dim=-1)
H_t = -(probs * torch.log(probs + 1e-12)).sum(dim=-1) / math.log(vocab_size)

if probs.max(...) > theta_safe:  # 极端确信，完全跳过
    final_logits = next_token_logits
else:
    # 熵门控
    g_vis = alpha_base_vis * torch.sigmoid((H_t - eta_vis) / tau_gate)
    g_txt = alpha_base_txt * torch.sigmoid((H_t - eta_txt) / tau_gate) \
            * (1 - math.exp(-gamma * time_step)) / math.exp(-gamma * time_step)

    # 自适应 cutoff
    beta_t = beta_base + beta_range * (1 - H_t)
    cutoff_th = torch.log(beta_t) + next_token_logits.max(dim=-1, keepdim=True).values
    next_token_logits = next_token_logits.masked_fill(next_token_logits < cutoff_th, -float("inf"))

    log_probs = torch.log_softmax(next_token_logits, dim=-1)
    log_probs_lang = torch.log_softmax(next_token_logits_lang_prior, dim=-1)
    log_probs_stat = torch.log_softmax(next_token_logits_stat_bias, dim=-1)

    # 门控对比解码
    final_logits = (1 + g_vis + g_txt) * log_probs \
                   - g_vis * log_probs_stat \
                   - g_txt * log_probs_lang
```

**修改 4: `methods/model_forward/` 和 `methods/utils/`**

无需修改。EntropyGate 复用 CRoPS 的 attention mask 操作（FastV 视觉 token 剪枝 + 文本 token 剪枝），这些构建幻觉模型的机制保持不变。

**修改 5: `run_entropygate.py`（新增，基于 run_crops.py）**

新增 EntropyGate 特有的命令行参数，其余评测流程完全复用。

### 2.7 计算开销分析

EntropyGate 相比 CRoPS 的额外计算开销极小：

| 操作 | 额外开销 |
|------|----------|
| 计算归一化熵 H_t | 一次 softmax + 一次 element-wise log-mul-sum，O(|V|) |
| 计算门控 g_t^{vis}, g_t^{txt} | 两次 sigmoid，O(1) |
| 自适应 cutoff | 一次标量运算，O(1) |

相比 CRoPS 每步 3 次完整前向传播的开销，这些额外计算可以忽略不计。

---

## 3. 实验设计 (Experimental Design)

### 3.1 实验模型

与 CRoPS 保持一致，在以下模型上评测：

- LLaVA-1.5-7B
- LLaVA-1.5-13B
- LLaVA-NeXT-8B
- Qwen2-VL-7B

### 3.2 基线方法

| 方法 | 类型 | 说明 |
|------|------|------|
| Sampling | 基础 | 标准 nucleus sampling |
| VCD | Training-free | 视觉对比解码 |
| ICD | Training-free | 指令对比解码 |
| OPERA | Training-free | 基于注意力惩罚 |
| ClearSight | Training-free | 注意力调整 |
| SID | Training-free | 自省解码 |
| M3ID | Training-free | 多模态互信息解码 |
| CRoPS | Training-free | 广义对比解码（我们的直接基线） |
| **EntropyGate** | **Training-free** | **本文方法** |

### 3.3 评测 Benchmark 与指标

#### 实验一：CHAIR Benchmark（物体级幻觉，对应 CRoPS Table 1）

在 MS-COCO 验证集上，使用 prompt "Please describe this image in detail" 生成描述。

CRoPS 论文原始结果（fp16，完整验证集）：

| Method | LLaVA-1.5 (7B) | | | LLaVA-1.5 (13B) | | | LLaVA-NeXT | | | Qwen2-VL | | |
|--------|----|----|----|----|----|----|----|----|----|----|----|----|
| | C_S↓ | C_I↓ | Recall↑ | C_S↓ | C_I↓ | Recall↑ | C_S↓ | C_I↓ | Recall↑ | C_S↓ | C_I↓ | Recall↑ |
| Sampling | 57.0 | 17.0 | 75.0 | 50.2 | 13.7 | 76.4 | 37.4 | 8.9 | 66.3 | 33.2 | 8.0 | 68.1 |
| ClearSight | 54.1 | 16.2 | 74.3 | 49.4 | 13.9 | 74.8 | 35.0 | 8.5 | 63.6 | 13.5 | 8.7 | 38.2 |
| VCD | 53.3 | 15.3 | 77.9 | 49.5 | 13.7 | 77.7 | 36.4 | 8.8 | 68.6 | 29.0 | 7.8 | 67.3 |
| ICD | 52.5 | 14.6 | 77.7 | 49.2 | 13.9 | 78.1 | 36.6 | 9.4 | 67.1 | 28.0 | 7.9 | 66.1 |
| OPERA | 49.1 | 13.8 | 78.5 | 48.2 | 13.2 | 78.9 | 35.5 | 8.9 | 66.9 | 31.0 | 8.1 | 67.9 |
| SID | 48.9 | 13.0 | 77.9 | 47.0 | 12.3 | 77.9 | 37.0 | 10.7 | 70.2 | 30.6 | 8.1 | 66.1 |
| M3ID | 47.1 | 12.8 | 74.8 | 45.5 | 12.2 | 75.3 | 36.1 | 9.8 | 68.7 | 28.8 | 7.3 | 64.8 |
| CRoPS | 39.5 | 10.2 | 76.3 | 38.5 | 9.1 | 75.1 | 33.2 | 8.1 | 66.2 | 26.9 | 6.9 | 67.4 |

我们的实验结果（4bit 量化，500 张子集）：

> 注：我们的实验使用 `load_in_4bit` 量化加载模型，并在 COCO val2014 的 500 张子集上评测（`chair_test_size=500`），因此数值与 CRoPS 论文原始结果（fp16，完整验证集）存在差异。

| Method | LLaVA-1.5 (7B, 4bit) | | |
|--------|----|----|-----|
| | C_S↓ | C_I↓ | Recall↑ |
| Sampling | 52.6 | 16.3 | 72.9 |
| CRoPS | 37.4 | 10.3 | 72.8 |
| **EntropyGate** | **49.0** | **12.5** | **77.6** |

预期效果：C_S 和 C_I 相比 CRoPS 进一步降低 5-10%，同时 Recall 保持或略有提升（因为低熵时减弱对比，避免过度抑制正确内容）。

#### 实验二：AMBER Benchmark（多维幻觉，对应 CRoPS Table 2）

使用 prompt "Please describe this image in detail"，评估物体幻觉 (CHAIR)、整体幻觉率 (HAL)、认知偏差 (Cog)。

| Method | LLaVA-1.5 (7B) | | | LLaVA-1.5 (13B) | | | LLaVA-NeXT | | | Qwen2-VL | | |
|--------|----|----|----|----|----|----|----|----|----|----|----|----|
| | CHAIR↓ | HAL↓ | Cog↓ | CHAIR↓ | HAL↓ | Cog↓ | CHAIR↓ | HAL↓ | Cog↓ | CHAIR↓ | HAL↓ | Cog↓ |
| Sampling | 10.6 | 44.3 | 4.0 | 9.3 | 41.3 | 4.2 | 10.5 | 57.1 | 4.1 | 6.4 | 38.9 | 2.8 |
| SID | 9.3 | 43.7 | 3.7 | 6.9 | 35.0 | 3.5 | 9.1 | 54.2 | 3.9 | 5.4 | 30.6 | 1.8 |
| M3ID | 9.0 | 40.0 | 3.0 | 7.9 | 40.0 | 2.9 | 8.7 | 51.9 | 3.1 | 5.5 | 27.9 | 1.5 |
| CRoPS | 6.3 | 29.3 | 2.8 | 5.7 | 27.8 | 2.5 | 7.2 | 44.6 | 2.6 | 5.1 | 24.2 | 1.1 |
| **EntropyGate** | **—** | **—** | **—** | **—** | **—** | **—** | **—** | **—** | **—** | **—** | **—** | **—** |

预期效果：HAL 和 Cog 进一步降低，尤其在认知偏差 (Cog) 上改善更明显——因为认知偏差往往发生在模型不确定但仍强行生成的时刻，EntropyGate 恰好在这些时刻加强视觉接地。

#### 实验三：POPE Benchmark（二元 VQA，对应 CRoPS Table 3）

| Method | LLaVA-1.5 | | LLaVA-NeXT | | Qwen2-VL | |
|--------|----|----|----|----|----|----|
| | Acc.↑ | F1↑ | Acc.↑ | F1↑ | Acc.↑ | F1↑ |
| Sampling | 81.7 | 82.8 | 87.3 | 87.6 | 84.2 | 82.8 |
| SID | 82.7 | 83.3 | 88.6 | 88.4 | 85.6 | 83.9 |
| M3ID | 82.4 | 83.4 | 88.0 | 88.1 | 84.9 | 83.5 |
| CRoPS | 83.9 | 84.6 | 89.4 | 89.4 | 86.1 | 85.3 |
| **EntropyGate** | **—** | **—** | **—** | **—** | **—** | **—** |

预期效果：POPE 是 yes/no 短回答任务，视觉 token 稀释效应较弱。EntropyGate 预期与 CRoPS 持平或略有提升（约 0.3-0.5%），因为短回答中熵变化不大。

#### 实验四：通用能力评测（对应 CRoPS Table 4）

| Method | LLaVA-1.5 | | LLaVA-NeXT | | Qwen2-VL | |
|--------|----|----|----|----|----|----|
| | MME↑ | MathVista↑ | MME↑ | MathVista↑ | MME↑ | MathVista↑ |
| Sampling | 1601 | 27.4 | 1669 | 34.8 | 2058 | 56.9 |
| SID | 1634 | 26.3 | 1607 | 36.8 | 2097 | 54.3 |
| M3ID | 1607 | 26.9 | 1683 | 36.6 | 2090 | 52.0 |
| CRoPS | 1662 | 28.9 | 1779 | 38.0 | 2184 | 55.6 |
| **EntropyGate** | **—** | **—** | **—** | **—** | **—** | **—** |

预期效果：通用能力不退化甚至略有提升。EntropyGate 在模型确信时减弱干预，理论上比 CRoPS 更少损害模型的原始能力。

#### 实验五：效率对比（对应 CRoPS Table 5）

在 NVIDIA A100 上，使用 LLaVA-1.5 7B 和 CHAIR benchmark 测量推理时间和显存。

| Method | Time (s)↓ | Memory (MB)↓ | C_S↓ |
|--------|-----------|--------------|------|
| Sampling | 215 | 15699 | 57.0 |
| Beam Search (5 beams) | 531 | 16737 | 50.7 |
| VCD | 550 | 17864 | 53.3 |
| SID | 510 | 16574 | 48.9 |
| OPERA | 1947 | 21943 | 49.1 |
| CRoPS | 652 | 16934 | 39.5 |
| **EntropyGate** | **—** | **—** | **—** |

预期效果：时间和显存与 CRoPS 几乎相同（额外的熵计算和门控运算开销可忽略），但 C_S 进一步降低。

### 3.4 消融实验

#### 消融一：门控机制的必要性

| 变体 | 说明 | C_S↓ | C_I↓ | Recall↑ |
|------|------|------|------|---------|
| CRoPS (baseline) | 固定 α^(1), 时间调度 α_t^(2) | 39.5 | 10.2 | 76.3 |
| EntropyGate (full) | 熵门控 g_t^{vis} + g_t^{txt} | — | — | — |
| w/o entropy gate | 仅保留时间调度，移除熵门控 (退化为 CRoPS) | — | — | — |
| w/o time modulation | 仅熵门控，移除 g_t^{txt} 中的时间递增项 | — | — | — |
| fixed gate (H_t=0.5) | 将 H_t 固定为 0.5，验证动态性的必要性 | — | — | — |

#### 消融二：熵阈值 η 的影响（对应 Laser Table 5 的设计思路）

| η_vis | η_txt | 平均触发率 | C_S↓ | C_I↓ | Recall↑ |
|-------|-------|-----------|------|------|---------|
| 0.1 | 0.2 | ~80% (几乎总是触发) | — | — | — |
| 0.2 | 0.3 | ~50% | — | — | — |
| 0.3 | 0.4 | ~30% (默认) | — | — | — |
| 0.4 | 0.5 | ~15% | — | — | — |
| 0.5 | 0.6 | ~5% (几乎不触发) | — | — | — |

预期趋势：阈值过低（总是触发）退化为 CRoPS；阈值过高（几乎不触发）退化为 Sampling。最优点在中间。

#### 消融三：门控温度 τ_gate 的影响

| τ_gate | 门控特性 | C_S↓ | C_I↓ | Recall↑ |
|--------|---------|------|------|---------|
| 0.01 | 近似硬门控 (0/1 切换) | — | — | — |
| 0.05 | 默认 (平滑 sigmoid) | — | — | — |
| 0.10 | 较软门控 | — | — | — |
| 0.20 | 非常软 (近似线性) | — | — | — |

#### 消融四：自适应 cutoff 的效果

| 变体 | C_S↓ | C_I↓ | Recall↑ |
|------|------|------|---------|
| 固定 β=0.1 (CRoPS 原始) | — | — | — |
| 自适应 β_t (EntropyGate) | — | — | — |

### 3.5 分析实验

#### 分析一：熵门控值随生成步数的变化

绘制 g_t^{vis} 和 g_t^{txt} 随生成步数 t 的变化曲线（类似 CRoPS Figure 2 左图的 VD(t) 曲线），对比：
- CRoPS 的固定 α^(1) 和启发式 α_t^(2)
- EntropyGate 的动态 g_t^{vis} 和 g_t^{txt}

预期观察：EntropyGate 的门控值呈现高度非均匀分布——在生成名词（物体名称）时出现峰值，在生成功能词时接近零。

#### 分析二：幻觉 token 与高熵 token 的相关性

统计分析：在 CHAIR benchmark 上，计算被标注为幻觉的物体名词 token 处的平均熵 vs 非幻觉 token 处的平均熵。

预期结果：幻觉 token 处的熵显著高于非幻觉 token，验证"高熵 = 高幻觉风险"的假设。

#### 分析三：共现偏差的缓解（对应 CRoPS Figure 2 右图）

对 ground truth 物体 "dining table" 的共现幻觉物体频率进行统计，对比 Sampling、M3ID、SID、CRoPS、EntropyGate。

预期结果：EntropyGate 在高熵时刻加强视觉对比，能更有效地抑制共现偏差。

#### 分析四：定性案例分析（对应 CRoPS Figure 3）

选取 3-5 个典型图像，展示：
- 各方法生成的描述（标红幻觉部分）
- EntropyGate 在每个 token 上的熵值和门控值热力图
- 说明 EntropyGate 如何在关键时刻（物体名称生成）加强视觉接地

---

## 4. 实现路线图

### Phase 1: 核心实现（基于 CRoPS 代码框架）

```
EntropyGate/
├── constants/
│   ├── crops_constants.py              # 复用
│   ├── entropygate_constants.py        # 新增：熵门控超参数
│   ├── default_generation_constants.py # 复用
│   └── image_token_constants.py        # 复用
├── methods/
│   ├── crops_method.py                 # 复用（patch 入口）
│   ├── entropygate_method.py           # 新增：EntropyGate patch 入口
│   ├── generation_configs/
│   │   └── contrastive_generation_config.py  # 扩展：新增 EntropyGate 参数
│   ├── model_forward/                  # 完全复用
│   │   ├── crops_llama_forward.py
│   │   └── crops_qwen_forward.py
│   ├── samplers/
│   │   ├── crops_sample.py             # 复用（作为基线）
│   │   └── entropygate_sample.py       # 新增：核心门控解码逻辑
│   └── utils/                          # 完全复用
│       ├── crops_forward_utils.py
│       └── crops_samplers_utils.py
├── benchmark/                          # 完全复用
├── utils/                              # 完全复用
├── run_crops.py                        # 复用（CRoPS 基线）
└── run_entropygate.py                  # 新增：EntropyGate 入口
```

### Phase 2: 评测流程

1. 先复现 CRoPS 在所有 benchmark 上的结果，确认代码框架正确
2. 运行 EntropyGate 在 CHAIR 上的实验，调优 η_vis, η_txt
3. 完成所有 benchmark 的完整评测
4. 运行消融实验
5. 运行分析实验，生成可视化

### Phase 3: 扩展实验（可选）

- 在更大模型（如 LLaVA-1.5-13B）上验证 scaling behavior
- 与 training-based 方法（如 RLHF-V）对比
- 探索将熵门控与 OPERA 的注意力惩罚结合

---

## 5. 预期贡献总结

1. **提出 token-level 熵门控机制**：首次将模型逐 token 的不确定性信号引入对比解码框架，实现视觉特征参与强度的自适应调节
2. **借鉴 Laser 的熵干预思想**：将 Laser 中用于潜空间推理的熵阈值机制，创造性地迁移到 training-free 幻觉治理场景
3. **保持 training-free 特性**：无需任何额外训练或外部模型，仅利用模型自身的输出分布熵作为门控信号
4. **与 CRoPS 框架无缝集成**：最小化代码修改，复用全部评测基础设施，便于公平对比

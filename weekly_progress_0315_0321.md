# 周报：EntropyGate — 基于熵门控的自适应对比解码抑制 VLM 幻觉

> 汇报人：李嘉琦 | 时间：2026-03-15 ~ 2026-03-21
> 模型：LLaVA-1.5-7B-HF | 主要评测：CHAIR (500 samples, COCO val2014)

---

## 一、相关工作

视觉语言模型 (VLM) 的幻觉问题是当前研究热点。现有的对比解码方法通过构造"幻觉分布"与原始分布做差来抑制幻觉，代表工作包括：

| 方法 | 会议 | 核心思路 |
|------|------|---------|
| VCD | CVPR 2024 | 对输入图像加高斯噪声构造对比信号 |
| DoLa | ICLR 2024 | 对比模型成熟层与早期层的 logits |
| ACD | EMNLP 2024 | 用熵比值 H_orig/(H_orig+H_c) 动态调整对比权重 |
| SID | ICLR 2025 | 注意力选择性扰动视觉 token |
| CRoPS | 2024 | 两步嵌套对比：语言先验 + 统计偏差视觉对比 |
| OPERA | CVPR 2024 | 注意力过度信任惩罚 + 回溯机制 |

**所有方法的共同局限**：对比强度 α 是固定或启发式设定的，与模型逐 token 的认知状态无关。

---

## 二、动机

通过实验观察到模型在逐 token 生成时的不确定性（熵）是高度动态的：

- **高确信 token**（低熵，如介词 "in"、"on"）：施加固定强度的对比解码会引入噪声，损害生成质量
- **低确信 token**（高熵，如判断物体是否存在时）：固定对比强度不够，无法有效抑制幻觉

固定 α 是一种 one-size-fits-all 的妥协。我们的核心想法是：**让对比强度根据每个 token 的熵动态调整**。

---

## 三、初始方案与 Baseline

### 3.1 Baseline

| 方法 | CHAIRs ↓ | CHAIRi ↓ | Recall ↑ |
|------|----------|----------|----------|
| Vanilla (无对比解码) | 52.6 | 16.3 | 72.9 |
| CRoPS (α=1.0, λ=0.01) | 37.4 | 10.3 | 72.8 |

### 3.2 初始 EntropyGate（Flat 结构）

初始设计使用**平坦公式**，将熵门控直接作用于视觉和文本两路对比信号：

```
g_vis = sigmoid((H_t - η_vis) / τ) * α_base_vis
g_txt = time_decay(t) * sigmoid((H_t - η_txt) / τ) * α_base_txt
final = (1 + g_vis + g_txt) * log_p - g_vis * log_p_stat - g_txt * log_p_lang
```

初始结果：CHAIRs=49.0, CHAIRi=12.5, Recall=77.6。Recall 很高，但幻觉抑制远不如 CRoPS。

---

## 四、实验过程与逐步改进

### 4.1 Scheme D — Floor-Gated（加下限）

发现原始 EntropyGate 的 g_vis 平均值仅 0.10（远低于 CRoPS 的固定 α=1.0），于是为 g_vis/g_txt 设置下限保证最低对比强度。

**最优 D-0.5/0.2**：CHAIRs=42.6（↓6.4 vs 初始），但仍不如 CRoPS。

### 4.2 Scheme D+A — Adaptive Eta 校准

在 D 基础上校准 η 阈值到实际熵分布，增大 α_base。

**最优 c1**：CHAIRs=42.2, CHAIRi=10.7（接近 CRoPS 的 10.3），但 CHAIRs 仍有 4.8 差距。

### 4.3 根因分析

对 CRoPS 和 EntropyGate 的公式做了等效系数展开分析，发现**三个核心差异**：

1. **公式结构差异（最关键）**：CRoPS 的两步嵌套结构产生交叉放大效应，原始分布权重是 EntropyGate 的 1.63 倍，lang prior 对比力度是 4.3 倍
2. **时间衰减启动太慢**：g_txt 前 20 步几乎为零
3. **Cutoff/Safety 阈值差异**：非核心但引入干扰

### 4.4 Scheme E — 嵌套结构 + 熵门控（关键突破）

基于根因分析，将 EntropyGate 嵌入 CRoPS 的嵌套结构中，熵门控只调制第二步（stat bias 对比）的强度：

```
intermediate = log_p + (1-γ_t)/γ_t * (log_p - log_p_lang)     # Step 1: 保留 CRoPS 原始 lang prior 结构
g_vis = α_min + (α_base - α_min) * sigmoid((H_t - η) / τ)     # 熵门控
final = (1 + g_vis) * intermediate - g_vis * log_p_stat        # Step 2: 自适应视觉对比
```

| 配置 | α_base_vis | α_min_vis | CHAIRs ↓ | CHAIRi ↓ | Recall ↑ |
|------|-----------|-----------|----------|----------|----------|
| E4 (固定 α=1.0, ≈CRoPS) | 1.0 | 1.0 | 37.0 | 10.1 | 73.2 |
| **E5 (熵门控 [0.5, 1.5])** | **1.5** | **0.5** | **35.6** | **9.9** | **73.6** |

### 4.5 其他探索

| 方案 | 思路 | 最优 CHAIRs | 结论 |
|------|------|------------|------|
| Scheme F (ACD 风格) | 用 H_orig/(H_orig+H_c) 作 α | 39.2 | Recall 好 (76.0) 但抑制力度不足 |
| Scheme G (对齐参数) | E + 对齐 cutoff/safety | 35.6 | 与 E 完全一致，说明非核心参数不是瓶颈 |
| Dir12/Dir34 (flat 调参) | flat 结构极限探索 | 40.2 | flat 结构天花板约 40，证实结构是瓶颈 |
| VCD + EntropyGate | 在 VCD 上加熵门控 | 55.8 (vs 58.4) | 有改善 (↓2.6) 但 VCD 路线整体弱于 CRoPS |

### 4.6 Latent Space 方向探索（新增）

在 logit 空间对比解码取得成果后，进一步探索将对比操作延伸到**隐状态空间 (hidden state space)**。核心动机：RMSNorm 是非线性操作，在 hidden state 空间做对比再投影回 logits，与直接在 logit 空间做对比的结果不同，可能带来额外收益。实现了三种方法：

#### HSC (Hidden State Contrastive) — 隐状态对比

将 stat bias 对比从 logit 空间移到 hidden state 空间，利用 RMSNorm 的非线性：

```
Stage 1: h_contrasted = h_orig + g_vis(H_t) * (h_orig - h_stat)    # 隐状态空间对比（熵门控）
         logits_corrected = lm_head(RMSNorm(h_contrasted))          # 非线性投影回 logits
Stage 2: final = log_p_corrected + time_mult * (log_p_corrected - log_p_lang)  # logit 空间 lang prior
```

实验配置 5 组 sweep：α_base ∈ {1.0, 1.5}，α_min ∈ {0.3, 0.5}，η ∈ {0.10, 0.30, 0.50}（含 LASER 启发的高 η 配置）。

#### LEG (Latent Entropy Gate) — 中间层熵门控

公式与 E5 完全相同（nested + 熵门控），但**熵信号来源不同**：从模型中间层（如第 16 层）的 hidden state 投影出 logits 再计算熵，而非使用输出层 logits 的熵。动机：中间层的不确定性可能更早地反映幻觉风险。

```
H_latent = entropy(lm_head(RMSNorm(h_mid_layer)))   # 中间层熵
g_vis = α_min + (α_base - α_min) * sigmoid((H_latent - η) / τ)
# 后续对比公式与 E5 一致
```

实验配置 5 组 sweep：hidden_layer ∈ {-8, -16, -24}（近输出层/中间层/早期层），含 LASER 风格高 η=0.30 配置。

#### LLH (Latent-Logit Hybrid) — 双重熵门控

两阶段混合：Stage 1 在隐状态空间做 stat bias 对比（由原始熵 H_t 门控），Stage 2 在 logit 空间做 lang prior 对比（由**修正后分布的熵 H_corrected** 门控，而非仅靠时间衰减）。实现**双重熵门控**。

```
Stage 1: h_contrasted = h_orig + g_vis(H_t) * (h_orig - h_stat)
Stage 2: H_corrected = entropy(lm_head(RMSNorm(h_contrasted)))
         g_txt = time_mult * sigmoid((H_corrected - η) / τ)
         final = (1 + g_txt) * log_p_corrected - g_txt * log_p_lang
```

实验配置 5 组 sweep：α_base ∈ {1.0, 1.5}，α_min ∈ {0.3, 0.5}，η ∈ {0.10, 0.30, 0.50}。

#### Latent 方向小结

三种方法均已完成代码实现（`methods/samplers/latent_sample.py`, `methods/utils/latent_utils.py`）、超参数定义（`constants/latent_constants.py`）和实验脚本（`scripts/run_hsc.sh`, `run_leg.sh`, `run_llh.sh`），每种方法配置了 5 组参数 sweep（含 LASER 启发的高 η 配置），待跑实验获取结果。这一方向将"对比操作在哪个空间进行"和"熵信号从哪一层获取"作为新的设计维度，进一步丰富了 EntropyGate 框架的通用性。

---

## 五、最终结果

| 方法 | CHAIRs ↓ | CHAIRi ↓ | Recall ↑ | vs CRoPS |
|------|----------|----------|----------|----------|
| Vanilla | 52.6 | 16.3 | 72.9 | — |
| CRoPS | 37.4 | 10.3 | 72.8 | baseline |
| **EntropyGate E5** | **35.6** | **9.9** | **73.6** | **CHAIRs -1.8, CHAIRi -0.4, Recall +0.8** |

E5 在三项指标上全面优于 CRoPS，且 E4（固定 α=1.0）的 CHAIRs=37.0 验证了嵌套结构的正确性，E5 vs E4 的 1.4 提升即为熵门控的纯贡献。

---

## 六、反思与下一步计划

### 6.1 反思

1. **公式结构比超参数更重要**：flat 结构无论怎么调参都无法突破 40，切换到 nested 后直接达到 35.6。前期在 flat 上花了较多时间调参，应更早做根因分析。
2. **4-bit 量化的意外发现**：fp16 下 nested CHAIRs=55.4，4-bit 下=35.6，差距高达 19.8。量化噪声可能增强了对比信号有效性，这一现象值得深入研究。
3. **通用性视角**：EntropyGate 的核心——"用熵门控替换固定 α"——理论上适用于所有对比解码方法（VCD、DoLa、SID 等），可定位为通用自适应层而非某个方法的 trick。
4. **从 logit 到 latent 的自然延伸**：对比操作不必局限在 logit 空间。RMSNorm 的非线性使得 hidden state 空间的对比有独立价值；中间层的熵信号可能比输出层更早捕捉到幻觉风险。三种 Latent 方法（HSC/LEG/LLH）将"对比空间"和"熵信号来源层"作为新的设计维度，丰富了框架的探索空间。

### 6.2 下一步

1. **运行 Latent 实验**：完成 HSC/LEG/LLH 共 15 组配置的 CHAIR 评测，与 E5 对比
2. **扩展基础方法**：在 VCD、DoLa 上验证 EntropyGate 的通用性
3. **扩展模型**：在 LLaVA-1.5-13B、Qwen2-VL 上验证
4. **扩展 Benchmark**：AMBER、POPE 等更多评测
5. **深入分析**：逐 token 熵-幻觉关系可视化，量化现象的机理探究，Latent 方向的 RMSNorm 非线性效应分析

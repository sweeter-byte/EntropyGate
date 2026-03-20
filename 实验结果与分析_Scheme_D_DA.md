# EntropyGate 实验结果与根因分析

> 实验日期：2026-03-16 ~ 2026-03-18
> 模型：LLaVA-1.5-7B-HF | 评测：CHAIR (500 samples, COCO val2014)

---

## 一、基线结果

### 1.1 4bit 量化

| 方法 | CHAIRs ↓ | CHAIRi ↓ | Recall ↑ |
|------|----------|----------|----------|
| Vanilla | 52.6 | 16.3 | 72.9 |
| CRoPS (α=1.0, λ=0.01) | 37.4 | 10.3 | 72.8 |
| 原始 EntropyGate | 49.0 | 12.5 | 77.6 |

### 1.2 fp16 精度

| 方法 | CHAIRs ↓ | CHAIRi ↓ | Recall ↑ |
|------|----------|----------|----------|
| Vanilla | 55.4 | 16.7 | 74.5 |
| CRoPS (α=1.0, λ=0.01) | 41.0 | 11.2 | 74.3 |
| 原始 EntropyGate | 47.4 | 12.3 | 77.8 |

### 1.3 原始 EntropyGate Gate 统计

| 指标 | 值 |
|------|-----|
| avg_H | 0.1212 |
| avg_g_vis | 0.0985 |
| avg_g_txt | 0.0184 |
| skip_rate | 16.6% |
| 样本数 | 136 (debug run) |

---

## 二、Scheme D 结果（Floor-Gated，5 组配置）

Scheme D 核心思想：为 g_vis 和 g_txt 设置下限（floor），保证最低对比强度。

| 配置 | floor_vis | floor_txt | CHAIRs ↓ | CHAIRi ↓ | Recall ↑ | avg_g_vis | avg_g_txt | skip_rate |
|------|-----------|-----------|----------|----------|----------|-----------|-----------|-----------|
| D-0.0/0.0 | 0.0 | 0.0 | 44.6 | 12.4 | 77.3 | 0.147 | 0.019 | 16.4% |
| D-0.3/0.1 | 0.3 | 0.1 | 44.0 | 11.7 | 76.6 | 0.367 | 0.089 | 16.7% |
| D-0.5/0.2 | 0.5 | 0.2 | **42.6** | **11.0** | 76.5 | 0.511 | 0.161 | 16.8% |
| D-0.7/0.3 | 0.7 | 0.3 | 43.8 | 11.2 | 75.8 | 0.657 | 0.232 | 16.9% |
| D-1.0/0.5 | 1.0 | 0.5 | 44.2 | 11.4 | 75.2 | 0.878 | 0.374 | 16.8% |

Scheme D 最优：D-0.5/0.2，CHAIRs=42.6, CHAIRi=11.0

---

## 三、Scheme D+A 结果（Floor + Adaptive Eta，6 组配置）

Scheme D+A 核心思想：在 Scheme D 基础上，将 eta 阈值校准到实际熵分布，并增大 alpha_base_vis 到 1.5。

| 配置 | floor_vis | floor_txt | eta_vis | eta_txt | tau | CHAIRs ↓ | CHAIRi ↓ | Recall ↑ | avg_g_vis | avg_g_txt | skip_rate |
|------|-----------|-----------|---------|---------|-----|----------|----------|----------|-----------|-----------|-----------|
| c1 | 0.5 | 0.2 | 0.10 | 0.15 | 0.05 | **42.2** | **10.7** | 75.3 | 0.921 | 0.401 | 17.0% |
| c2 | 0.5 | 0.2 | 0.12 | 0.18 | 0.05 | 43.2 | 11.0 | 75.4 | 0.870 | 0.355 | 16.9% |
| c3 | 0.7 | 0.3 | 0.08 | 0.12 | 0.05 | 44.2 | 11.4 | 74.9 | 1.025 | 0.485 | 17.0% |
| c4 | 0.5 | 0.2 | 0.10 | 0.15 | 0.10 | 43.2 | 11.0 | 75.0 | 0.897 | 0.413 | 16.9% |
| c5 | 0.7 | 0.3 | 0.12 | 0.18 | 0.10 | 43.4 | 11.1 | 75.0 | 0.939 | 0.422 | 17.0% |
| c6 | 1.0 | 0.5 | 0.12 | 0.18 | 0.05 | 43.6 | 11.1 | 75.2 | 1.057 | 0.483 | 17.0% |

Scheme D+A 最优：c1，CHAIRs=42.2, CHAIRi=10.7

---

## 四、结果对比总览

| 方法 | CHAIRs ↓ | CHAIRi ↓ | Recall ↑ | avg_g_vis | avg_g_txt | skip_rate |
|------|----------|----------|----------|-----------|-----------|-----------|
| Vanilla | 52.6 | 16.3 | 72.9 | — | — | — |
| CRoPS (α=1.0, λ=0.01) | **37.4** | **10.3** | 72.8 | 1.0 (fixed) | ~0.01 (fixed) | — |
| 原始 EntropyGate | 49.0 | 12.5 | **77.6** | 0.10 | 0.02 | 16.6% |
| Scheme D best (0.5/0.2) | 42.6 | 11.0 | 76.5 | 0.51 | 0.16 | 16.8% |
| Scheme D+A best (c1) | 42.2 | 10.7 | 75.3 | 0.92 | 0.40 | 17.0% |

关键发现：
- D+A c1 的 CHAIRi (10.7) 已接近 CRoPS (10.3)，差距仅 0.4
- 但 CHAIRs (42.2 vs 37.4) 仍有 4.8 的差距
- EntropyGate 系列的 Recall 始终高于 CRoPS（75.3 vs 72.8）

---

## 五、根因分析：EntropyGate vs CRoPS 差距

### 5.1 差异一：公式结构不同（最关键）

CRoPS 的公式是**两步嵌套**的：
```
step1: intermediate = log_p/γ - (1-γ)/γ * log_p_lang
step2: final = (1+α) * intermediate - α * log_p_stat
```

展开后等效系数：
```
coeff_orig = (1+α)/γ
coeff_lang = (1+α)*(1-γ)/γ
coeff_stat = α
```

EntropyGate 的公式是**平坦的**：
```
final = (1+g_vis+g_txt) * log_p - g_vis * log_p_stat - g_txt * log_p_lang
```

关键区别：CRoPS 中 `(1+α)` 乘在 intermediate 上，stat bias 的去除是在已经放大过的信号上操作的。

在平均 112 步的生成中：
- CRoPS 平均：coeff_orig=3.71, coeff_lang=1.71, coeff_stat=1.0
- EntropyGate D+A c1 平均：coeff_orig=2.32, coeff_lang=0.40, coeff_stat=0.92

**EntropyGate 的原始分布权重只有 CRoPS 的 63%，lang prior 对比力度只有 CRoPS 的 23%。**

### 5.2 差异二：g_txt 的时间衰减启动太慢

EntropyGate 沿用了 CRoPS 的 `(1-γ)/γ` 时间衰减（γ=exp(-0.01*t)），但应用方式不同：

| step | γ_t | time_mult | EntropyGate g_txt | CRoPS lang系数 |
|------|-----|-----------|-------------------|----------------|
| 1 | 0.990 | 0.010 | 0.005 | 1.000 |
| 5 | 0.951 | 0.051 | 0.024 | 1.000 |
| 10 | 0.905 | 0.105 | 0.050 | 1.000 |
| 20 | 0.819 | 0.221 | 0.105 | 1.000 |
| 50 | 0.607 | 0.649 | 0.307 | 1.000 |
| 100 | 0.368 | 1.718 | 0.812 | 1.000 |

前 20 步 g_txt 极小，文本对比信号几乎不存在。而 CRoPS 的 stat bias 系数始终是 1.0，不受时间衰减影响。

### 5.3 差异三：Cutoff 和 Safety 阈值差异

| 参数 | CRoPS | EntropyGate |
|------|-------|-------------|
| cutoff beta | 0.10 (固定) | 0.18 (自适应，更宽松) |
| safety skip | max_prob > 0.95 | max_prob > 0.99 |

- EntropyGate 的 beta 更大 → 保留更多候选 token → 对比解码不够聚焦
- EntropyGate 的 theta_safe=0.99 → 几乎不跳过（17%），而 CRoPS 的 0.95 阈值会跳过更多高确信 token

---

## 六、CRoPS 等效系数分析

CRoPS 公式展开后，在 γ=exp(-0.01*t), α=1.0 条件下：

| step | γ_t | coeff_orig=(1+α)/γ | coeff_lang=(1+α)(1-γ)/γ | coeff_stat=α |
|------|-----|---------------------|--------------------------|--------------|
| 1 | 0.990 | 2.020 | 0.020 | 1.0 |
| 10 | 0.905 | 2.210 | 0.210 | 1.0 |
| 50 | 0.607 | 3.297 | 1.297 | 1.0 |
| 100 | 0.368 | 5.437 | 3.437 | 1.0 |
| 112 (avg) | 0.326 | 6.132 | 4.132 | 1.0 |

平均 112 步：coeff_orig ≈ 3.71, coeff_lang ≈ 1.71, coeff_stat = 1.0

这意味着 CRoPS 在后期对原始分布的放大倍数远超 EntropyGate，且 lang prior 的对比力度随时间大幅增长。

---

## 七、改进方向

### 方向 1：对齐公式结构（推荐优先尝试）

采用 CRoPS 的两步嵌套结构，让 entropy gate 只控制各步的强度：
```python
intermediate = log_p + g_txt_time * (log_p - log_p_lang)
final = (1 + g_vis) * intermediate - g_vis * log_p_stat
```
其中 `g_txt_time` 保留 `(1-γ)/γ` 的时间衰减，`g_vis` 由 entropy gate 控制。

### 方向 2：将 gamma_decay 从 g_txt 中移除

让 g_txt 不再乘以时间衰减，或者大幅增大 gamma_decay（比如 0.1），让衰减在前几步就快速收敛。

### 方向 3：对齐 cutoff 和 safety 阈值

将 theta_safe 降到 0.95，beta 固定为 0.1，先消除这些非核心差异的干扰。

### 方向 4：最简方案

在 CRoPS 的原始公式上，只用 entropy gate 来调制 `α_stat_bias`，即 `α = g_vis(H_t)`，其他保持不变。这样改动最小，最容易超过 CRoPS。

---

## 八、文献调研与新改进方案（2026-03-18 更新）

### 8.1 相关工作调研

| 方法 | 会议 | 核心机制 | 对 EntropyGate 的启示 |
|------|------|----------|----------------------|
| DoLa | ICLR 2024 | 层间对比，JSD 动态选层 | 熵可替代 JSD 作为层选择信号 |
| VCD | 2023 | 视觉噪声对比，固定 alpha | alpha 固定是局限 |
| SID | ICLR 2025 | 注意力选择性扰动视觉 token | 精细化扰动优于粗糙噪声 |
| ACD | EMNLP 2024 | **熵比值自适应 alpha** | 最直接相关：用 H_orig/(H_orig+H_c) 作为动态权重 |
| OPERA | CVPR 2024 | 注意力过度信任惩罚+回溯 | 注意力模式是幻觉先行指标 |
| 精确无损抑制 | 2025 | JS 散度选择性激活对比 | gate 不应始终开启 |
| VaLiD | 2024 | 熵加权多层视觉融合 | 高熵层包含更多需纠正信息 |
| Epic | 2025 | k 步前瞻熵重加权 | 前瞻熵比当前熵更稳定 |

### 8.2 核心洞察

1. **ACD (EMNLP 2024)** 的熵比值机制最值得借鉴：`alpha = H(Y_t) / (H(Y_t) + H(Y_t^c))`。当对比模型也不确定时，说明对比信号本身不可靠，应自动减弱。EntropyGate 目前只看原始模型的 H_t，忽略了对比模型的不确定性。

2. **精确无损语言先验抑制 (2025)** 的选择性激活思想：只在检测到语言先验主导时才启动对比解码，避免过度抑制损害召回。

3. **嵌套结构的数学优势**：CRoPS 的两步嵌套天然产生交叉放大效应，g_vis=1.0 时等效系数完全匹配 CRoPS。

### 8.3 新改进方案

#### 方案 E：嵌套结构 + 熵门控（推荐优先实验）

```python
# Step 1: 去除 lang prior (保留 CRoPS 原始时间衰减结构)
gamma_t = exp(-gamma_decay * t)
intermediate = log_p + (1-gamma_t)/gamma_t * (log_p - log_p_lang)

# Step 2: 去除 stat bias (由熵门控控制强度)
g_vis = alpha_min_vis + (alpha_base_vis - alpha_min_vis) * sigmoid((H_t - eta_vis) / tau)
final = (1 + g_vis) * intermediate - g_vis * log_p_stat
```

等效系数分析（gamma_decay=0.01, 平均 112 步）：

| g_vis | avg_c_orig | avg_c_lang | c_stat | vs CRoPS |
|-------|------------|------------|--------|----------|
| 0.5 | 2.779 | 1.279 | 0.500 | 75% |
| 0.8 | 3.335 | 1.535 | 0.800 | 90% |
| 1.0 | 3.706 | 1.706 | 1.000 | 100% |
| 1.2 | 4.076 | 1.876 | 1.200 | 110% |

优势：
- g_vis=1.0 时数学上等价于 CRoPS，保证下限
- 熵门控只调制 stat bias 强度，不破坏 lang prior 的时间衰减结构
- c_orig 和 c_lang 自动获得交叉放大，无需手动增大 alpha_base
- 建议 sweep: g_vis ∈ {0.8, 1.0, 1.2}，配合 floor=0.5

#### 方案 F：ACD 风格熵比值门控

```python
# 计算三个分布的熵
H_orig = entropy(softmax(logits_orig))
H_stat = entropy(softmax(logits_stat_bias))
H_lang = entropy(softmax(logits_lang_prior))

# ACD 风格: 对比模型越确定 → 对比信号越可靠 → 权重越大
alpha_vis = H_orig / (H_orig + H_stat + eps)
alpha_txt = H_orig / (H_orig + H_lang + eps)

# 结合嵌套结构
intermediate = log_p + alpha_txt * (1-gamma_t)/gamma_t * (log_p - log_p_lang)
final = (1 + alpha_vis) * intermediate - alpha_vis * log_p_stat
```

优势：
- 无需手动调 eta, tau, alpha_base 等超参数
- 自动适应不同 token 的对比需求
- 当对比模型也不确定时自动减弱（避免引入噪声）
- 理论上更优雅，但需要验证实际效果

#### 方案 G：对齐非核心参数 + 嵌套结构

在方案 E 基础上，同时对齐 cutoff 和 safety 参数：
- theta_safe: 0.99 → 0.95
- beta: 自适应 → 固定 0.1
- gamma_decay: 保持 0.01

这样可以隔离"公式结构"这一变量的影响。

### 8.4 建议实验顺序

1. **方案 E** (嵌套+熵门控): 改动最小，数学上有保证，sweep g_vis ∈ {0.8, 1.0, 1.2}
2. **方案 G** (E + 对齐参数): 在 E 的最优配置上对齐 cutoff/safety
3. **方案 F** (ACD 熵比值): 如果 E/G 效果好，尝试用熵比值替代固定 gate，追求更优雅的自适应

# 周报：EntropyGate — 熵门控自适应对比解码缓解 VLM 幻觉

> 汇报人：李嘉琦 | 时间：2026-03-15 ~ 2026-03-21
> 模型：LLaVA-1.5-7B-HF | 评测：CHAIR (500 samples, COCO val2014)

---

## 一、相关工作

VLM 的对象幻觉问题近年来受到广泛关注，对比解码是其中一类主流方案。其基本思路是构造一个"容易产生幻觉"的辅助分布，然后在解码时从原始分布中减去该辅助分布的影响。按照辅助分布的构造方式，代表方法如下：

| 方法 | 出处 | 辅助分布构造 |
|------|------|-------------|
| VCD | CVPR 2024 | 对输入图像加高斯噪声 |
| DoLa | ICLR 2024 | 取模型早期层的 logits |
| ACD | EMNLP 2024 | 与 VCD 类似，但用 H\_orig/(H\_orig+H\_c) 做自适应权重 |
| SID | ICLR 2025 | 选择性扰动注意力中的视觉 token |
| CRoPS | 2024 | 两步嵌套：先去语言先验，再去统计偏差 |
| OPERA | CVPR 2024 | 基于注意力的过度信任惩罚 + 回溯 |
| LEAD | CVPR 2026 | 高熵时改用概率加权连续嵌入 + 视觉锚点 |
| LASER | 2026 | 隐状态叠加态 + 熵正则化干预 |

这些方法普遍存在的一个问题是：对比强度 α 要么是手动设定的常数，要么依赖简单的衰减策略，没有考虑模型在生成每个 token 时的实际不确定性。

---

## 二、出发点

在跑 baseline 实验的过程中，我注意到一个现象：模型对不同 token 的确信程度差异很大。比如生成介词 "in""on" 时，模型的输出熵很低，这种时候如果仍然施加较强的对比修正，反而会在本来就正确的预测上引入噪声；而在判断某个物体是否存在时，模型熵较高，此时固定的 α 又可能不够用。

这就引出了一个很自然的想法：能不能让对比强度跟着当前 token 的熵走？熵高时加大力度、熵低时减小干预。

---

## 三、初始方案与 Baseline

### 3.1 Baseline

| 方法 | CHAIRs ↓ | CHAIRi ↓ | Recall ↑ |
|------|----------|----------|----------|
| Vanilla | 52.6 | 16.3 | 72.9 |
| CRoPS (α=1.0, λ=0.01) | 37.4 | 10.3 | 72.8 |

### 3.2 初始 EntropyGate（Flat 结构）

最初的设计比较直接，把熵门控以平坦的方式加在视觉和文本两路对比信号上：

$$g_{\text{vis}} = \sigma\!\bigl((H_t - \eta_{\text{vis}})/\tau\bigr) \cdot \alpha_{\text{base,vis}}$$

$$g_{\text{txt}} = \text{decay}(t) \cdot \sigma\!\bigl((H_t - \eta_{\text{txt}})/\tau\bigr) \cdot \alpha_{\text{base,txt}}$$

$$\text{final} = (1 + g_{\text{vis}} + g_{\text{txt}}) \log p - g_{\text{vis}} \log p_{\text{stat}} - g_{\text{txt}} \log p_{\text{lang}}$$

跑出来 CHAIRs=49.0, CHAIRi=12.5, Recall=77.6。Recall 倒是不错，但幻觉抑制效果比 CRoPS 差了一大截。

---

## 四、实验过程与改进

### 4.1 Scheme D — 加 Floor 下限

看了一下 gate 的统计数据，发现 g\_vis 的平均值才 0.10，而 CRoPS 相当于 α 恒为 1.0。对比强度差了一个数量级，效果差也就不奇怪了。于是给 g\_vis 和 g\_txt 加了一个下限（floor），保证最低对比强度。

最优配置 D-0.5/0.2：CHAIRs=42.6（比初始版降了 6.4），但跟 CRoPS 的 37.4 还有距离。

### 4.2 Scheme D+A — 校准 η 阈值

在 D 的基础上把 η 校准到实际的熵分布范围，同时把 α\_base 拉大到 1.5。

最优 c1：CHAIRs=42.2, CHAIRi=10.7。CHAIRi 已经接近 CRoPS（10.3），但 CHAIRs 还差 4.8。

### 4.3 根因分析

到这一步我意识到光调超参数可能不够，于是把 CRoPS 和 EntropyGate 的公式做了展开对比。CRoPS 的两步嵌套结构是：

$$\text{intermediate} = \frac{\log p}{\gamma} - \frac{1-\gamma}{\gamma} \log p_{\text{lang}}$$

$$\text{final} = (1+\alpha) \cdot \text{intermediate} - \alpha \cdot \log p_{\text{stat}}$$

展开后可以看到，原始分布前面的系数是 $(1+\alpha)/\gamma$，随时间步增长会变得很大。而我们 flat 公式里原始分布的系数只有 $1 + g_{\text{vis}} + g_{\text{txt}}$，数值上只有 CRoPS 的 63\% 左右。lang prior 的对比力度更是只有 CRoPS 的 23\%。

结论很明确：**公式结构本身就有问题**，嵌套结构天然带来的交叉放大效应，flat 结构怎么调参都模拟不出来。

### 4.4 Scheme E — 嵌套 + 熵门控（主要结果）

想清楚之后，做法就很自然了：直接复用 CRoPS 的嵌套结构，只用熵门控替换第二步里固定的 α：

$$\text{intermediate} = \log p + \frac{1-\gamma_t}{\gamma_t}(\log p - \log p_{\text{lang}})$$

$$g_{\text{vis}} = \alpha_{\min} + (\alpha_{\text{base}} - \alpha_{\min}) \cdot \sigma\!\!\left(\frac{H_t - \eta}{\tau}\right)$$

$$\text{final} = (1 + g_{\text{vis}}) \cdot \text{intermediate} - g_{\text{vis}} \cdot \log p_{\text{stat}}$$

| 配置 | α\_base | α\_min | CHAIRs ↓ | CHAIRi ↓ | Recall ↑ |
|------|---------|--------|----------|----------|----------|
| E4（固定 α=1.0, ≈CRoPS） | 1.0 | 1.0 | 37.0 | 10.1 | 73.2 |
| **E5（熵门控 [0.5, 1.5]）** | **1.5** | **0.5** | **35.6** | **9.9** | **73.6** |

E4 把 α 固定为 1.0，等价于 CRoPS，跑出来 37.0 和 CRoPS 的 37.4 基本一致，说明嵌套结构对接没有问题。E5 再加上熵门控，CHAIRs 进一步降到 35.6，这 1.4 的提升就是熵门控本身带来的。

### 4.5 其他探索

| 方案 | 做法 | 最优 CHAIRs | 简评 |
|------|------|------------|------|
| Scheme F（ACD 风格） | 用 H\_orig/(H\_orig+H\_c) 做 α | 39.2 | Recall 较好 (76.0) 但幻觉指标不如 E5 |
| Scheme G（参数对齐） | E + 对齐 cutoff/safety | 35.6 | 跟 E 一样，说明这些参数不是瓶颈 |
| Dir12/Dir34（flat 极限） | flat 下各种调参 | 40.2 | 证实 flat 结构天花板约 40 |
| VCD + EntropyGate | VCD 上加熵门控 | 55.8 (vs 58.4) | 降了 2.6，但 VCD 整体不如 CRoPS |

### 4.6 Latent Space 方向（新实现）

logit 空间的实验告一段落后，我开始尝试把对比操作搬到隐状态空间。主要想法是：模型最后一层到 logits 之间有个 RMSNorm，它是非线性的，所以在 hidden state 空间做对比再投影回来，跟直接在 logit 空间做对比得到的结果是不一样的。实现了三种方法：

**HSC (Hidden State Contrastive)**：把 stat bias 对比搬到 hidden state 空间，lang prior 对比留在 logit 空间。

$$\mathbf{h}_{\text{contrasted}} = \mathbf{h}_{\text{orig}} + g_{\text{vis}}(H_t) \cdot (\mathbf{h}_{\text{orig}} - \mathbf{h}_{\text{stat}})$$

$$\text{logits}_{\text{corrected}} = W_{\text{head}} \cdot \text{RMSNorm}(\mathbf{h}_{\text{contrasted}})$$

然后在 logit 空间做 lang prior 对比（带时间衰减）。

**LEG (Latent Entropy Gate)**：对比公式跟 E5 一样在 logit 空间做，但驱动门控的熵信号改成从中间层（比如第 16 层）取。想法是中间层的不确定性可能比输出层更早反映幻觉风险。

**LLH (Latent-Logit Hybrid)**：两阶段都用熵门控——第一阶段在 hidden state 空间做 stat bias 对比（用原始熵 $H_t$ 门控），第二阶段在 logit 空间做 lang prior 对比（用修正后分布的熵 $H_{\text{corrected}}$ 门控）。

三种方法的代码和实验脚本都已写好，各配了 5 组参数（包括参考 LASER 的高 η 配置），待跑实验。

---

## 五、当前最优结果

| 方法 | CHAIRs ↓ | CHAIRi ↓ | Recall ↑ | vs CRoPS |
|------|----------|----------|----------|----------|
| Vanilla | 52.6 | 16.3 | 72.9 | — |
| CRoPS | 37.4 | 10.3 | 72.8 | baseline |
| **EntropyGate E5** | **35.6** | **9.9** | **73.6** | **-1.8 / -0.4 / +0.8** |

三项指标全面优于 CRoPS。

---

## 六、新论文调研与方法构想

### 6.1 LEAD (CVPR 2026)

论文全称 *Thinking in Uncertainty: Mitigating Hallucinations in MLRMs with Latent Entropy-Aware Decoding*（arXiv: 2603.13366）。

LEAD 观察到转折词（because, however 等）常处于高熵状态，幻觉容易在这之后出现。它的做法有两点：一是高熵时不选具体 token，改用所有 token embedding 的概率加权混合作为下一步输入：

$$\tilde{\mathbf{e}}_t = \begin{cases} \mathbf{e}(r_t) & H_t < \hat{H} \\ \mathbb{E}_{v \sim p_t}[\mathbf{e}(v)] & H_t \geq \hat{H} \end{cases}$$

二是高熵时注入视觉锚点：$\tilde{\mathbf{e}}_t^* = (1-\lambda)\,\mathbb{E}_{v \sim p_t^*}[\mathbf{e}(v)] + \lambda\,\mathbf{e}_{\text{vis}}$，λ=0.4 效果最好。

在 R1-Onevision-7B 上，MMHalu +4.7\%, VStar +4.7\%, MathVista +2.3\%。

### 6.2 LASER (arXiv 2601.06803)

LASER 认为 CoT 的显式文本推理存在信息瓶颈——连续的视觉信息在 token 化时会丢失。它的做法是在隐状态空间保持"语义叠加态"，用动态窗口上的 soft distribution 做监督，并在模型不确定时注入 hard target：

$$P_t^{\text{target}} = \begin{cases} \alpha \cdot y_{\text{hard}} + (1-\alpha) \cdot Q_t & H(Q_t) > \eta \\ Q_t & H(Q_t) \leq \eta \end{cases}$$

其中 η=0.6，约 10\% 的 token 会触发干预。HallusionBench 上提升 11.36\%，推理 token 量减少 97.3\%。

### 6.3 LEAD 作为 baseline 的可行性

对比了一下 CRoPS 和 LEAD 的定位，它们解决问题的角度差别很大：

| | CRoPS | LEAD |
|---|-------|------|
| 干预方式 | 修改 logits 分布 | 切换输入 embedding |
| 干预时机 | 每个 token | 仅高熵 token |
| 目标场景 | 通用 VLM | 推理模型（R1 系列） |
| 假设 | 幻觉来自视觉/语言偏差 | 幻觉来自高熵点的错误坍缩 |

**不太适合直接替代 CRoPS**，但可以作为另一路 baseline 做横向对比。两者可以正交组合。

### 6.4 新方法构想

结合 LEAD 的视觉锚点、LASER 的叠加态思想和我们的隐状态对比框架，想到三个方向：

**方案 A — SGCD（叠加态引导的对比解码）**：高熵 token 上，先把 h\_orig 替换成概率加权的叠加态表示，再做对比。好处是叠加态保留了多个候选语义。问题是从 embedding 空间到 hidden state 空间的投影开销不小。

**方案 B — VALC（视觉锚点隐状态对比）**：在 HSC 的基础上加一项视觉锚点。提取视觉 token 的平均 hidden state $\mathbf{h}_{\text{vis}}$，高熵时把 $\mathbf{h}_{\text{orig}}$ 往 $\mathbf{h}_{\text{vis}}$ 方向拉：

$$\mathbf{h}_{\text{out}} = \mathbf{h}_{\text{orig}} + g_{\text{vis}} \cdot (\mathbf{h}_{\text{orig}} - \mathbf{h}_{\text{stat}}) + \lambda(H_t) \cdot (\mathbf{h}_{\text{vis}} - \mathbf{h}_{\text{orig}})$$

对比项去掉幻觉成分，锚点项拉回视觉信息，改动量极小（HSC 加一行代码）。**我倾向于先做这个。**

**方案 C — EAEF（自适应嵌入反馈）**：对比解码修正 logits 后，高熵时用修正后分布的概率加权 embedding 作为下一步输入（不走 argmax）。好处是修正后的分布更干净，连续嵌入又保留了语义多样性。难点是要改 embedding 输入逻辑，侵入性较大。

| 优先级 | 方案 | 改动量 | 额外开销 |
|--------|------|--------|---------|
| 1 | B (VALC) | 极小 | 无 |
| 2 | C (EAEF) | 中等 | embedding 矩阵乘 |
| 3 | A (SGCD) | 大 | 可能要额外 forward |

---

## 七、反思与下一步

### 7.1 反思

1. 前期在 flat 结构上花了不少时间调超参数，回头看应该更早去分析公式结构本身。flat 怎么调都过不了 40，换成 nested 一下子到了 35.6，说明**结构比参数重要**。
2. 一个没预料到的现象：fp16 下 nested CHAIRs=55.4，4-bit 量化下是 35.6，差了将近 20 个点。猜测是量化引入的噪声反而让对比信号更有效，但具体机理还说不清楚。
3. "用熵门控替换固定 α"这个想法本身不局限于 CRoPS，理论上 VCD、DoLa、SID 上都能用，可以定位为对比解码的通用自适应层。
4. 从 logit 到 latent 的推进比较自然——RMSNorm 的非线性让 hidden state 空间的对比有独立意义，中间层的熵信号也可能比输出层更早反映幻觉风险。

### 7.2 下一步

1. 跑 HSC/LEG/LLH 共 15 组实验，跟 E5 对比
2. 实现 VALC（方案 B），验证视觉锚点 + 对比解码的协同效果
3. 在同一 benchmark 上跑 LEAD，做横向对比
4. 在 VCD、DoLa 上验证 EntropyGate 的通用性
5. 扩模型（13B、Qwen2-VL）和 benchmark（AMBER、POPE）
6. 逐 token 的熵-幻觉可视化分析，4-bit 量化现象的机理探究

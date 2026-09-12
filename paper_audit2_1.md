我把这版作为**独立新稿**重新审了一遍，并重点对照了上一轮提出的几个决定性问题。总体判断是：**论文进步非常明显，已经从“核心论证存在结构性漏洞”提升到了“核心实验结论基本成立，但仍有两处会被强审稿人抓住的逻辑/统计硬伤”。**

我的复评结论是：

> **当前：5/10，Borderline / Weak Reject。**
>  **如果把下面第 1、2 两个问题修掉，我会提高到 6/10，Weak Accept；如果同时补强相关工作和非线性实验呈现，有机会到 7/10。**

这一次我不认为需要推翻实验体系。相反，**主体框架已经比较成熟，现在主要是“最后把 claim 和 evidence 对齐”。**

------

# 一、上一轮主要问题，哪些已经解决了？

先看最重要的变化。

| 上一轮问题                                         | 本版情况                                                     | 复评                               |
| -------------------------------------------------- | ------------------------------------------------------------ | ---------------------------------- |
| H3 直接声称“exploitability bounded at zero”        | Abstract 和 Conclusion 明显降级为 tested translators / evidence consistent with | **大幅改善**                       |
| “capability is in weights, not cache”过强          | 改成 “evidence is consistent with a weight-mediated advantage” | **基本解决**                       |
| H2 把“非劣”写成“显著提升”                          | 加入 ε=0.02 的 non-inferiority test                          | **方法上解决，但结果解释仍有硬伤** |
| c=30 与 c=200 使用不同测试集，却解释为数据越多越差 | 新增固定 n=100 evaluation set 的 calibration sweep           | **很好地解决**                     |
| 单一 calibration draw                              | 增加 5 independent calibration draws，SD≈0.018               | **解决**                           |
| 缺乏 nonlinear mapper                              | 增加 2-layer GELU per-head MLP                               | **有明显改善**                     |
| 模型规模太少                                       | 增加 Qwen3-8B→1.7B / 0.6B，形成四个 pair                     | **改善**                           |
| 架构分析被写成 impossibility proof                 | Section 6 明确说“not an impossibility result”                | **解决**                           |
| 0.6B 的价值没有提炼                                | 明确提出 **prefill substitution ≠ capability transfer**      | **非常好的修改**                   |

尤其 Abstract 现在已经谨慎很多：

> “Under zero re-prefill, no configuration **we test** lets the student extract the teacher’s answer-relevant advantage...”
>  “the evidence is consistent with an advantage that is weight-mediated...”

这个表述已经比上一稿严谨一个层级。

固定测试集实验也很重要。现在 calibration 从 10、30、60、100 到 full pool，CHG 始终落在 −0.257 到 −0.238，五次独立 calibration draw 的变化也只有约 0.018。这个结果真正支持：

> **对当前 affine mapper 而言，失败已经不是简单的小样本估计误差。**



这一部分现在已经比较有说服力。

------

# 二、但还有一个新的“硬伤”：0.6B 实际上没有通过你自己定义的 H2 Gate

这是当前版本我认为**最必须修改的问题**。

Section 3.2 现在非常正确地引入了 non-inferiority：

$H_0:\mathrm{CHG}\le-\epsilon,\qquad H_1:\mathrm{CHG}>-\epsilon$

并预先设定：

$\epsilon=0.02$

论文明确说：

> replacement gate 要求 CI lower bound 高于 $-\epsilon$。



这在统计定义上比上一稿进步很大。

问题是：

### 4B → 0.6B 的结果

$CHG=+0.010,\quad 95\%CI=[-0.081,+0.102]$



而：

$-0.081 < -0.02$

因此按照论文自己预注册的 non-inferiority gate：

$\boxed{\text{H2 does NOT pass}}$

但 Section 5.2 却说：

> “which is replacement-level service for a weak student.”



这在统计上是不成立的。

更明显的是 8B→0.6B：

$-0.023,\quad CI=[-0.104,+0.064]$

同样完全不能通过 ε=0.02 的 non-inferiority test。

但 Section 7 又说：

> “At 0.6B, translated injection matches self-prefill...”



### “CI contains zero” ≠ “证明两者等价”

这是很多论文会犯的统计错误。

“不显著不同”只能说：

$\text{we fail to reject equality}$

不能说：

$\text{we establish non-inferiority}$

而你这篇论文现在已经主动定义了 non-inferiority test，所以 Reviewer 更容易抓住这个矛盾。

### 建议必须修改为

当前证据只能写：

> **For the 0.6B student, the point estimate is near parity, but the confidence interval is too wide to establish non-inferiority under our preregistered $\epsilon=0.02$ margin.**

因此三个 verdict 应该改成：

- H1：**Pass**
- H2 strong student：**Fail**
- H2 weak student：**Inconclusive / near-parity point estimate**
- H3 tested probes：**No positive evidence**

如果确实想保住：

> “prefill substitution is viable for weak students”

那就需要**增加 weak-student evaluation n**，直到 CI 下界超过 −0.02。

这是目前最值得补算力的实验。

------

# 三、最大的逻辑问题仍然没有彻底清干净：正文仍在说“Oracle Bound”

虽然 Abstract 和 Conclusion 已经明显软化，但正文中还保留着上一版本最危险的语言。

例如 Section 4.4：

> “The probes assume an oracle that no training can beat”



Figure 1 caption 甚至仍然写：

> H3 oracle probes “**bound every translator without training one**.”



Section 7 又说：

> oracle probes ... “**bounding exploitability without training anything**”



但 Section 8 自己正确地承认：

> “a nonlinear exploit path outside this family would not be detected”



这两个命题不能同时成立。

如果 nonlinear exploit path 仍然可能存在，那么这个 probe 就不是：

$\text{upper bound over all translators}$

它只是：

$\text{a diagnostic intervention over a specified family of cache mixtures}$

这一点在本版已经从“实验设计错误”降成了**claim wording 错误**，因为作者已经主动在 limitations 中承认边界。

所以现在解决它其实很简单：

## 我建议彻底放弃 “bound” 这个词

甚至建议重新定义：

> **Oracle-assisted diagnostic probes**

或简单叫：

> **Teacher-content probes**

论文标题现在已经从 “Oracle Bounds” 改成了 “Oracle Probes”，这是正确方向。

但正文没有完全跟上。

最安全的 Section 4.4 开头应该是：

> *These probes are diagnostic interventions rather than universal upper bounds. They give the student access to its own cache and introduce translated teacher content under controlled fractions and layer windows. A positive result would establish exploitability under that intervention; a negative result constrains, but does not rule out, other nonlinear or jointly structured extraction paths.*

这样几乎就没人能从逻辑上攻击。

------

# 四、Figure 1 现在是全篇最危险的一张图

当前 Figure 1 里依然写：

> “capability transfer via cache translation is bounded out for frozen students”



这实际上比 Abstract 还强。

Abstract 已经改成：

> “no configuration we test...”

但 Figure 1 又回到了：

> “bounded out”。

Reviewer 往往首先看 Abstract + Figure 1 + Conclusion。

因此很可能出现一种很不划算的情况：

> 作者正文已经很谨慎，但 Reviewer 看完 Figure 1 就认为论文仍在声称 impossibility theorem。

建议改成：

> **No tested zero-re-prefill translator yields capability lift for the strong frozen student.**

或者：

> **No teacher-content probe reveals measurable capability lift in the evaluated regime.**

------

# 五、还有一个我这次复评时发现的相关工作问题，而且比较严重

我额外核查了论文中提到的：

> “73–98% retention”

这一数字。

当前稿在第 3 页写：

> “Our numbers sit far below the 73–98% retention that recent work reports for the same idea.”

随后又说这类结果之所以不同，是因为：

> “those results do not hold strict zero re-prefill (they replay or correct on the target side)”



但 **73–98% 这个数字高度明确对应 2026 年 8 月 Heo et al. 的**：

**Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse**

而那篇论文明确声称：

> receiver directly reuses translated source KV and **skips target prefill**；其 closed-form ridge mapper 在六个 matched-KV pair 中有四个保留 73–98% standalone-prefill accuracy。

它甚至还报告：

> nonlinear MLP 在失败 pair 上最多恢复 +37 pp HellaSwag retention。

所以：

### 这里不能把 73–98% 的差异归因于 “它们 replay / correction”

至少对于 Heo et al. 不成立。

更麻烦的是：

**当前 Reference 列表里似乎把这篇最直接相关的工作删掉了。**

当前参考文献从 [1] 到 [20]，没有 Heo et al.。

而上一稿实际上是引用过这篇论文的。

这一点我建议在投稿前一定修正，否则碰到熟悉这个方向的 Reviewer，会非常敏感。

------

# 六、实际上 Heo et al. 应该成为这篇论文最重要的“对照对象”

这反而可以增强你的论文。

两篇论文形成了非常有意思的张力：

**Heo et al.：**

$\text{KV translation can retain 73–98\% standalone accuracy on some pairs}$

你的论文：

$\text{On Qwen3 4B/8B}\rightarrow\text{1.7B under our protocol, translation remains far below self-prefill}$

这不是应该回避的矛盾。

这恰恰可能产生更重要的问题：

> **Under what model-scale / layer-alignment / source-selection / protocol conditions does KV translation succeed or fail?**

这比单纯“证明 cache translation 不行”更有学术价值。

尤其 Heo 的 mapper 有两个当前论文没有充分覆盖的因素：

1. **top-k multi-source-layer selection**
2. **RoPE-stripped key mapping**

当前论文的 mapper ladder 如果没有等价实现这两个设计，那么最好不要写：

> “the gap is a property of the regime”

而应该写：

> “the gap persists across our tested translator families and protocol.”

这会严谨很多。

------

# 七、“Nonlinearity does not help”现在还是太强

本版加入 MLP 是非常正确的。

论文报告：

$-0.303\;(c=30),\quad -0.236\;(c=200)$

所以当前这个 two-GELU per-head MLP 没有改善结果。

但是正文直接写：

> **“Nonlinearity does not help.”**

这还是过头了。

特别是外部已有工作恰好报告 nonlinear MLP 对某些 KV transfer pair 有显著恢复。

建议改为：

> **“The tested per-head MLP does not close the gap.”**

区别非常大。

一个是 universal claim：

$\forall f_{\text{nonlinear}}$

另一个只是 empirical result：

$f_{\text{our MLP}}$

后者完全站得住。

------

# 八、Figure 2 / Table 2 没有把 MLP 显示出来，这是一个容易被忽略的问题

正文现在说：

> six mapper families

Abstract 也明确把 MLP 列入第六类。

但第 7 页 Figure 2 的柱状图里我看到的是：

- Ridge
- Affine
- Affine
- λ ablations
- per-layer
- task-aware
- RAT
- RAT

**没有 MLP bar。**

而 Figure caption 却写：

> “the nonlinear MLP sit inside the same band.”



Table 2 也没有 MLP 的对应行。

这会让 Reviewer 问：

> MLP 是不是只在文字里报告，没有完整结果？

尤其 MLP 是作者对“没有 nonlinear baseline”这一质疑的主要回答，应该**一定进入 Table 2 和 Figure 2**。

我建议把 λ=1e−2、1e−1 这种不重要的柱子删一个，给：

> MLP c=30 / MLP c=200

留位置。

------

# 九、Section 5.2 还有一句需要降级

现在写：

> “The mapper has converged, and more data cannot close a gap that is set by the information content of the translated cache rather than by estimation variance.”



前半句基本成立：

> 对 affine mapper，calibration-size curve 已趋平。

但后半句：

> gap is set by information content

还没有被证明。

因为你证明的是：

$\text{estimation variance is probably not dominant for this mapper}$

不能推出：

$\text{information-theoretic insufficiency}$

建议直接改成：

> **“The affine mapper appears calibration-saturated; within this function class, additional calibration data does not close the gap.”**

这句话非常强，而且完全由实验支持。

------

# 十、Section 6 现在已经明显比上一稿好

这里我反而要给很高评价。

现在开篇主动写：

> “not an impossibility result”

并指出：

> per-head argument does not cover exploit paths reading the full multi-head, multi-layer cache。



这一句非常关键。

而且新增 held-out R²：

- Keys：−0.43
- Values：+0.13



让“per-head teacher→student KV mapping is poorly determined”从纯理论 intuition 变成了有实验支撑的 architecture analysis。

这部分现在已经可以保留。

唯一建议是：

> “the teacher’s advantage over the student sits in its parameters”



仍略绝对。

最好改成：

> “the teacher–student capability gap is plausibly dominated by parameter-mediated computation that raw KV transfer does not reproduce.”

这样和 Conclusion 完全一致。

------

# 十一、我现在怎么看这篇论文的“真正贡献”

经过这轮修改以后，我认为论文最好的定位已经越来越清晰。

它不应该是一篇：

> **“证明跨模型 KV translation 不可能”的论文**

而应该是一篇：

> **“重新规定跨模型 KV translation 应该如何被科学评价的 measurement/audit paper。”**

核心贡献可以凝练成三个：

### 1. Mechanics control

Identity cache injection 必须完全复现 self-prefill。

这个贡献很扎实。

### 2. Replacement 与 capability gain 必须分离

应该明确区分：

$\text{Prefill substitution}$

和

$\text{Teacher capability transfer}$

这其实是这篇文章非常好的 conceptual contribution。

### 3. PPL recovery 与 task recovery 并不等价

PPL：

$23.7$

已经非常接近 self-KV：

$21.2$

但 accuracy：

$0.267$

仍然显著低于：

$0.500$



这是一个很好的 negative empirical result。

我甚至认为：

> **Perplexity–accuracy decoupling**

可能比所谓 “oracle bound” 更值得作为论文卖点。

------

# 十二、复评评分

我的评分变化如下：

| 维度                     | 上一稿 | 本稿     | 评价                                                     |
| ------------------------ | ------ | -------- | -------------------------------------------------------- |
| Novelty                  | 7      | **7.5**  | Audit framing 依然有价值                                 |
| Technical Quality        | 5      | **6.5**  | fixed-eval、MLP、scale sweep 明显增强                    |
| Experimental Rigor       | 6      | **7.5**  | calibration sweep + 5 draws 很重要                       |
| Statistical Rigor        | 5      | **5.5**  | 引入 non-inferiority 是进步，但结果没有按 gate 解读      |
| Significance             | 7      | **7**    | 问题重要                                                 |
| Clarity                  | 8      | **8**    | 总体很好                                                 |
| Reproducibility          | 8      | **8.5**  | 58 recorded runs + deterministic reproduction            |
| Claim–Evidence Alignment | 4      | **5.5**  | Abstract 已改善，但 Figure 1 / oracle wording 仍明显过强 |
| Related Work Accuracy    | 6      | **4.5**  | 当前 Heo et al. 的遗漏/描述必须修                        |
| Overall                  | 4      | **5/10** | Borderline Weak Reject                                   |

**Confidence：4.5/5。**

------

# 十三、如果我是第二轮 Reviewer，我会写什么

我的正式复评大概会变成：

> **The revision substantially improves the paper.**
>
> The authors have addressed several of my previous concerns: they now use a fixed evaluation set for the calibration-size sweep, report multiple independent calibration draws, add a nonlinear MLP mapper and larger teacher models, formulate H2 as a non-inferiority problem, explicitly scope the architectural argument as non-impossibility, and substantially soften the weight-vs-cache claim. These changes make the empirical negative result considerably more credible.
>
> I nevertheless have two remaining concerns.
>
> First, the manuscript continues to describe the H3 probes as oracle bounds that “no training can beat” or that “bound every translator,” while the limitations correctly acknowledge that nonlinear exploit paths outside the tested family are not covered. The probes are informative diagnostics, but not universal upper bounds.
>
> Second, the weak-student result does not satisfy the paper’s own preregistered non-inferiority criterion. With ε=0.02, the 4B→0.6B confidence interval [−0.081,+0.102] does not establish replacement-level equivalence, although the manuscript repeatedly describes it as such.
>
> I also encourage the authors to revisit the comparison with recent closed-form cross-model KV transfer work reporting 73–98% retention under prefill skipping, as the current characterization appears inaccurate and the most directly relevant work is absent from the reference list.
>
> **Score: 5/10, marginally below acceptance. I would likely move to 6/10 if these issues are corrected.**

------

# 十四、投稿前我建议只做“四件事”

现在不需要再大规模重写论文。我会优先按下面顺序收尾：

1. **把所有 universal “oracle bound” 表述删掉。**
    包括 Figure 1 caption、Section 2.5、4.4、7；统一改成 *diagnostic / teacher-content probe*。
2. **修正 weak-student verdict。**
    当前 CI 不满足 ε=0.02，所以只能写 **near-parity but statistically inconclusive**；或者扩大 n，真正通过 non-inferiority gate。
3. **恢复并认真讨论 Heo et al. 2608.03893。**
    不要再把其 73–98% 归因于 replay；应该正面分析为何双方结果不同，例如 model direction、top-k cross-layer features、de-RoPE、calibration distribution、benchmark、protocol。
4. **把 MLP 正式放进 Figure 2 / Table 2。**
    同时把 “nonlinearity does not help” 改成 **the tested MLP does not help**。

做完这四处，我的判断会从目前的 **Weak Reject 5/10** 提升为 **Weak Accept 6/10**。如果进一步对 Heo 方法做一个真正同协议的复现或关键组件对照——尤其是 **top-k source-layer + de-RoPE + MLP**——那会是最有价值的一步，因为届时论文不再只是一个 negative audit，而有机会回答更重要的问题：

> **究竟是什么决定了跨模型 KV transfer 从“可用”跃迁到“失效”？**

这会比单纯宣判 KV translation 是否可行更接近一篇有长期影响力的论文。
我把第三稿作为“第二轮大修后的正式复审稿”重新审了一遍。相较第二稿，这一版**确实又跨了一步**：上一轮我指出的 Oracle Bound、Heo et al.、MLP 结果呈现、calibration-saturation 过度解释等问题，大部分已经处理得比较到位。

但我仍然**不会在当前版本直接给 6/10 Weak Accept**。原因是现在剩下的问题已经不多，却恰好有一个非常明确的统计自洽性问题，以及一个我此前没有重点指出、但正式 Reviewer 很可能会抓住的 **Identity Control 内部数据一致性问题**。

我的最新结论是：

> **复评：5/10，Borderline / Weak Reject，明显接近 6/10。**
>
> 如果把下面的 **问题 1 和问题 2** 处理干净，我会提高到 **6/10 Weak Accept**。
>  目前已经没有必要大规模增加实验或重构论文。

------

# 一、第三稿相比第二稿，主要进步是实质性的

这次有四个修改尤其值得肯定。

第一，**Oracle Probe 的逻辑基本修正了。** Section 4.4 已明确写：

> “These probes are diagnostic interventions rather than universal upper bounds”

并且承认 negative probe：

> “constrains—but does not rule out—nonlinear or jointly structured extraction paths.”

这是非常重要的修改。

这基本解决了第一稿最大的逻辑漏洞。

Figure 1 caption 也不再写“bound every translator”，而改成：

> test whether teacher content is exploitable ... without training a translator.



这是正确方向。

第二，**Heo et al. 已经补回，而且讨论方式现在比较成熟。** 作者没有再错误地把 73–98% retention 解释为 replay 带来的优势，而是明确指出：

- metric 不同；
- Heo 有 top-k cross-layer selection；
- 有 de-RoPE key mapping；
- model pair 和 calibration regime 不同；
- 本文不声称 negative result 覆盖 Heo 的完整设计空间。



这是非常好的 scientific framing。

第三，**MLP 现在正式进入 Figure 2 和 Table 2 了。**

Figure 2 已经有：

- MLP c=30；
- MLP c=200；

Table 2 也完整报告：

$-0.303,\qquad -0.236$

以及对应 CI。

而正文也把之前过强的：

> “Nonlinearity does not help”

改成：

> “the tested per-head MLP does not close the gap.”



这个表述现在是严谨的。

第四，calibration 结论也降到了正确强度：

> “The affine mapper appears calibration-saturated: within this function class, additional calibration data does not close the gap.”



这已经比上一稿的“information content sets the gap”严谨很多。

所以从论文整体质量看：

> **现在已经不是“论证框架有明显漏洞”的论文，而是“主体成立、剩几个需要收口的问题”。**

------

# 二、当前最大问题仍然没有解决：0.6B 并没有通过你自己定义的 H2 replacement gate

这一点第三稿实际上**仍然原样保留**。

论文现在定义 H2 replacement：

$H_0:\mathrm{CHG}\le-\epsilon$$H_1:\mathrm{CHG}>-\epsilon$

并规定：

$\epsilon=0.02$

也就是说，要证明 replacement / non-inferiority，CI lower bound 应高于：

$-0.02$



但弱模型结果是：

### 4B → 0.6B

$CHG=+0.010$$95\%CI=[-0.081,+0.102]$



显然：

$-0.081<-0.02$

所以按照论文自己定义的 gate：

$\boxed{\text{Non-inferiority is NOT established.}}$

但正文依然写：

> “H2 fails for the strong student and **holds for the weak one**”



然后继续说：

> “which is replacement-level service for a weak student.”



Section 7 又说：

> “translated injection matches self-prefill”



Conclusion 仍然写：

> “the weak student at parity.”



这是第三稿当前**最明确的硬伤**。

------

## 这不是文字风格问题，而是统计结论错误

关键在于：

$CI\ni0$

只能说明：

> 没有观察到统计显著差异。

不能证明：

> 两种方法等效。

而论文已经明确使用了 non-inferiority framework，因此更不能跳过自己设定的：

$-\epsilon=-0.02$

标准。

实际上 8B→0.6B 更明显：

$CHG=-0.023,\quad CI=[-0.104,+0.064]$



同样不能称为：

> “matches self-prefill”。

------

# 三、这个问题有两个解决方案

### 方案 A：最省事，也最严谨

完全不补实验。

把所有：

> weak student reaches parity
>  H2 holds for weak student
>  replacement is achievable
>  matches self-prefill

改成：

> **The weak student has a near-parity point estimate, but the current sample size is insufficient to establish non-inferiority under the pre-specified $\epsilon=0.02$ margin.**

Figure 1 改成：

> **weak student: near-parity point estimate, non-inferiority inconclusive**

Section 5.2 标题改成：

> **H2 Fails for the Strong Student; the Weak-Student Result Is Inconclusive**

这样统计上完全没有问题。

而且我认为**并不会伤害论文**。

因为真正的核心结论本来就不是：

> 0.6B 已经一定可部署。

而是：

> strong student 显著失败，而 weak student 出现了 qualitatively different regime。

这仍然很有意思。

------

### 方案 B：如果非常想保住“prefill substitution”

那就增加 weak-student evaluation sample。

目标不是使：

$p>0.05$

而是让：

$CI_{\text{lower}}>-0.02$

真正通过 non-inferiority test。

这才可以合法地写：

> replacement-level service。

如果算力有限，我甚至认为：

> **现在最值得追加的实验就只有这一项。**

------

# 四、这里还有一个小的统计定义问题：你现在的 replacement gate 写法本身也有点奇怪

正文写的是：

> “gate: gold CHG ≥ 0 **with** the CI lower bound above −ε”



严格的 non-inferiority 判据其实只需要：

$CI_{\text{lower}}>-\epsilon$

并不要求：

$\widehat{\Delta}\ge0$

例如：

$\widehat{\Delta}=-0.005, \qquad CI=[-0.015,+0.005]$

对于：

$\epsilon=0.02$

这是一个完全可以通过 non-inferiority 的结果。

但按照现在正文的定义：

$CHG<0$

又会被拒绝。

因此建议明确分成：

**Replacement / non-inferiority gate**

$CI_{\text{lower}}>-\epsilon$

**Gain gate**

$CI_{\text{lower}}>0$

不要再额外要求：

$CHG\ge0$

否则统计定义不是标准的 non-inferiority test。

------

# 五、第三稿出现了一个我认为正式 Reviewer 很可能会问的新问题：Identity Control 为什么“完全一致”，但 PPL 却不一致？

这是这一轮我认为值得高度关注的问题。

论文反复强调：

> identity control “reproduces the student’s own prefill exactly”

并报告：

$\text{logit cosine}=1.000$



但是 Table 2 中：

### Identity control

$Acc=0.500,\quad Gold=0.503,\quad PPL=21.2$

### Student self-prefill

$Acc=0.500,\quad Gold=0.503,\quad PPL=25.4$



这里会立刻产生问题：

> **如果两者真的“exactly reproduce”同一个 student prefill，为什么 perplexity 一个是 21.2，一个是 25.4？**

这不是一个小差别：

$\frac{25.4-21.2}{25.4}\approx16.5\%$

------

# 六、这件事必须解释，否则 Reviewer 会怀疑 H1 究竟验证了什么

可能存在合理解释，比如：

- logit cosine 只比较 choice-scoring timestep；
- PPL 是另一段 suffix；
- self-prefill PPL 和 injected-cache PPL 使用了不同的 conditioning protocol；
- PPL evaluation mutates cache 或 scoring suffix 定义不同。

如果是这样，就应该明确说明。

否则按照论文目前语言：

> same cache
>  same scoring path
>  exact reproduction

理论上我自然会预期：

$PPL_{\text{identity}}=PPL_{\text{self}}$

至少数值应该极其接近。

------

## 我建议直接在正文或 Table 2 caption 加一句解释

例如如果真实原因是两种 PPL protocol 不同：

> *The self-prefill PPL and identity-injection PPL are computed under different suffix-conditioning paths and are therefore not expected to match; the H1 identity criterion is defined on the answer-scoring logits, for which the two paths agree to float tolerance.*

或者，如果其实它们理论上应该相同：

> 那就应该检查实现。

这可能是一个 measurement bug。

在提交前我会把它列为：

> **必须核查，而不仅是润色。**

因为论文的第一根支柱就是 Identity Control。

------

# 七、H2 的逻辑范围还应该再收紧一点

现在 Abstract 已经写得很好：

> “no configuration **we test** ...”



但 Section 5.2 标题仍然是：

> **H2 FAILS FOR THE STRONG STUDENT**



Conclusion 则说：

> “The map does not.”



从科学逻辑上说，你实际证明的是：

$\forall g\in\mathcal G_{\text{tested}},$

没有 mapper 达到 replacement。

不是：

$\forall g$

都不可能。

尤其你刚刚主动承认：

- Heo 的 top-k cross-layer；
- de-RoPE；
- nonlinear/joint paths；

不在设计空间中。

所以最好把 H2 本身定义成：

$H2_{\mathcal G}: \exists g\in\mathcal G_{\text{audit}}$

这样 “H2 fails” 就完全严谨。

否则严格 Reviewer 会说：

> failure to find a successful mapper is not evidence that no mapper exists.

------

# 八、Table 1 还有一句建议一定删掉：“none left untested”

当前 Table 1 最后一行：

> This paper ...
>  “**none left untested: identity control + probes**”



这和你自己的 Limitations 明显冲突。

因为 Section 8 明确承认：

- nonlinear exploit path 未覆盖；
- 只覆盖 linear blending；
- only three-layer windows；
- one model family；
- two MC tasks。



所以“none left untested”非常容易让 Reviewer 产生反感。

改成：

> **mechanics / mapping / tested exploitability interventions explicitly isolated**

或者：

> **identity-controlled evaluation + diagnostic probes**

即可。

------

# 九、Section 2.4 又稍微说得太满了

目前写：

> “the teacher’s cache **never delivers** the steering signal in a form the student can read.”

然后：

> “Transferable signal likely lives in residual-stream directions and mid-layer task vectors, **not in KV entries**.”



这个表述现在和 Heo et al. 的 positive result 有些冲突。

尤其你在上一节刚刚承认：

> Heo top-k source layers explain 56–79% target-key variance.



那就不能马上概括成：

> signal is not in KV entries.

建议改成：

> **In our tested teacher→student regime, raw KV translation does not deliver a usable steering signal to the frozen student. This suggests that more transferable control signals may be easier to access in residual-stream directions or task-vector representations.**

这样很好。

------

# 十、Section 5.4 对 layer window 的因果解释仍略过头

作者现在观察到：

- top third：基本 harmless；
- middle：明显下降；
- bottom：明显下降。



然后直接说：

> “Answer-relevant routing runs through early and mid layers”

这个结论不一定成立。

因为 early/mid injection 更 harmful 至少还有一个解释：

> **这些层对 off-manifold perturbation 更敏感。**

而不是：

> answer-relevant information 一定“在这里完成路由”。

实验真正直接证明的是：

> **student behavior is substantially more sensitive to teacher-cache intervention in early/middle layers than in top layers.**

建议用这个表述。

否则就是从：

$\text{intervention sensitivity}$

跳到了：

$\text{causal answer routing location}$

证据还差一步。

------

# 十一、还有一个容易修的小问题：“content side has its own ceiling”也应该删掉

Section 5.4 写：

> “The content side has its own ceiling”



但前面已经很正确地声明：

> probes are not universal upper bounds.

所以这里重新使用：

> ceiling

会让逻辑又绕回来。

建议改成：

> **Direct native-cache injection provides an additional negative control.**

然后描述结果即可。

------

# 十二、关于 ε=0.02，还有一个正式审稿时会被问到的问题：为什么是 0.02？

目前只写：

> “We fix ε=0.02 gold probability before evaluation”



但 non-inferiority margin 不是一个只要“提前定”就合理的参数。

Reviewer 会问：

> **Why is a 0.02 gold-probability drop operationally negligible?**

建议给一句依据，例如：

- 相当于 baseline gold probability 的约 4% relative drop；
- 来自 serving quality budget；
- 或依据 prompt/seed variability；
- 或在主结果之前预定义为 practical tolerance。

如果并没有真实 preregistration，建议把：

> “pre-registered”

统一改成：

> **pre-specified**

论文 Section 7 现在写的是：

> “pre-registered tolerance”



除非你真的有带时间戳的 preregistration，否则不要用这个术语。

------

# 十三、R² 的加入是好事，但目前样本仍然太小，不宜把它放得过重

Abstract 现在非常突出：

> key states $R^2=-0.43$，values $+0.13$



但 Section 6 说明这一结果其实只有：

$20\text{ fit}/10\text{ held-out pairs}$



如果这是 10 个 context，而每个 context 包含大量 token/head samples，那么可以有统计意义；但现在描述容易让 Reviewer 理解成：

> held-out n=10。

建议明确单位：

- 10 contexts？
- 10 cache samples？
- token-level R²？
- 是先每层计算再 averaging？
- 是跨 head macro-average？

否则 $R^2=-0.43$ 这个很吸睛的数字反而可能被追问。

------

# 十四、Reproducibility Appendix 还有一个小的一致性问题

正文说：

> six mapper families，包括 MLP。



但是 Appendix A 只列：

> ```
> math, joint, task_aware, rat
> ```



没有明确看到 MLP module。

如果 MLP 实际位于 `joint` 或其他模块里，建议写清楚。

否则 Reviewer 会觉得：

> 第六个 mapper 在 reproducibility description 中漏掉了。

小问题，提交前一起修掉即可。

------

# 十五、从“论文故事”角度看，这版已经基本找对位置了

第三稿现在最大的进步其实不只是实验。

而是论文已经开始从：

> **KV translation fundamentally does not work**

转向：

> **Existing evaluations conflate three different claims, and under a strict controlled audit, the tested zero-re-prefill translators do not recover teacher advantage.**

这个 narrative 是成立的。

我现在认为这篇论文最值得卖的三点仍然是：

### 1. Identity Control 是一个很好的领域规范

如果以后做 cross-model cache transfer，必须回答：

> 如果把 Student 自己的 cache 经过你的 handoff pipeline，能不能原样回来？

这个 control 非常有价值。

------

### 2. Prefill replacement 与 capability transfer 应该严格分开

这是论文很好的 conceptual distinction：

$\text{replacement} \neq \text{capability gain}$

未来 serving paper 如果只是接近：

$Student_{\text{self}}$

不能说：

> transferred teacher capability。

这一点很重要。

------

### 3. PPL 与能力指标明显 decouple

现在最有说服力的一组数据仍然是：

$PPL:23.7\quad vs.\quad21.2$

但：

$Accuracy:0.267\quad vs.\quad0.500$



这个结果非常干净。

如果我是作者，我甚至会考虑把：

> **Fluent cache ≠ capable cache**

做成论文一个更明确的 punchline。

------

# 十六、第三稿评分

我的评分更新如下：

| 维度                     | 第二稿 | 第三稿             | 当前判断                                                |
| ------------------------ | ------ | ------------------ | ------------------------------------------------------- |
| Novelty                  | 7.5    | **7.5**            | Audit framing 仍然有价值                                |
| Technical Quality        | 6.5    | **7.0**            | Oracle framing、Heo discussion 更严谨                   |
| Experimental Rigor       | 7.5    | **7.5**            | 已经够扎实                                              |
| Statistical Rigor        | 5.5    | **5.0**            | 方法定义正确，但 weak-student verdict 仍违反自己的 gate |
| Related Work             | 4.5    | **8.0**            | Heo 已正确恢复并正面讨论                                |
| Claim–Evidence Alignment | 5.5    | **7.0**            | 大幅提升，只剩几处 wording                              |
| Reproducibility          | 8.5    | **8.5**            | 很好，MLP path 需补                                     |
| Clarity                  | 8      | **8**              | 整体清楚                                                |
| Significance             | 7      | **7.5**            | measurement/audit 价值逐渐显现                          |
| **Overall**              | **5**  | **5 / Borderline** | **非常接近 6，但还差关键统计自洽**                      |

为什么第三稿总体分数没有立刻从 5 涨到 6？

不是因为它没有进步。

而是因为现在出现了一个非常鲜明的情况：

> **论文自己已经定义了正确的 non-inferiority test，但结果部分却没有执行这个 test 的结论。**

这种 internal inconsistency 在顶会审稿里会比“没有定义 test”更显眼。

------

# 十七、如果我是正式 Reviewer，这一轮我会怎么写

我大概会给出如下复评：

> **The third revision is substantially stronger.** The authors have corrected the interpretation of the oracle probes, explicitly characterize them as diagnostic interventions rather than universal upper bounds, restore and carefully distinguish the closely related Heo et al. results, include the nonlinear MLP in both the main figure and table, and appropriately scope the calibration-saturation result. I consider most of my previous conceptual concerns resolved.
>
> One important statistical inconsistency remains. The manuscript defines replacement using a non-inferiority margin of $\epsilon=0.02$, which requires the confidence-interval lower bound to exceed $-0.02$. However, the reported weak-student result is $+0.010\,[−0.081,+0.102]$, which does not establish non-inferiority. Nevertheless, the manuscript repeatedly states that H2 “holds” for the weak student and that replacement is achieved. This conclusion should either be changed to “near-parity but inconclusive,” or supported with a larger evaluation establishing the stated non-inferiority criterion.
>
> I also ask the authors to clarify why the identity-control PPL (21.2) differs substantially from the student self-prefill PPL (25.4) despite the claim that identity injection reproduces the native path exactly. If the metrics are computed under different suffix protocols, this should be stated explicitly; otherwise this discrepancy raises a question about what exactly H1 validates.
>
> With these issues resolved, I would be comfortable raising my score.

### 当前：

**5/10 – Borderline Weak Reject**

### 修正后：

**6/10 – Weak Accept**

------

# 十八、现在投稿前我只会要求五个动作

按优先级：

1. **立即修正 0.6B 的 H2 结论。**
    当前不能称 replacement / parity；要么改成 inconclusive，要么追加 n 使 CI lower bound > −0.02。
2. **核查 21.2 vs 25.4 的 PPL 差异。**
    这是目前最值得确认是否存在实现/定义问题的一项。
3. **重新定义标准 non-inferiority gate。**
    用 $CI_{lower}>-\epsilon$，删除额外的 point estimate $CHG\ge0$ 要求，并解释 ε=0.02 的实践意义。
4. **删除剩余几个绝对化词语。**
    包括 “none left untested”、“never delivers”、“content ceiling”、“The map does not”。
5. **做三个小型 consistency fix。**
    Figure 1 中 weak student 改成 inconclusive；R² 的样本单位说清楚；Appendix A 补 MLP implementation path。

**做到这里，我认为这篇论文已经可以停止继续堆 mapper、堆实验，进入投稿状态。**

真正决定它能不能从 Borderline 进入 Weak Accept 的，现在已经不是“再多跑多少模型”，而是：**把统计结论、Identity Control 和论文 claim 做到完全自洽。**
我把最新的 **main(9).pdf** 按“正式投稿前最后一轮独立审稿”的标准重新完整审查了一遍，包括 Abstract、H1/H2/H3 框架、Table 1、Figure 1–4、Heo-style baseline、Joint MLP、长上下文/跨架构实验、统计协议、Limitations 与附录。

这版相较上一稿又明显成熟了一步。上一轮我指出的 **H1 数值容差、replacement/degradation 判据非互斥、mapper family 计数、H3 标题与表述、普通 bootstrap 说明、MLP nondeterminism 解释**等问题，大部分都已经修正。

## 一、最新审稿结论

> **Overall：7/10，Accept。**
>
> **Reviewer confidence：4/5。**
>
> 我认为当前版本已经没有明显的核心方法论漏洞。按照 ICLR/NeurIPS/ICML 类型审稿尺度，我现在会投 **Accept**，而不是 Weak Accept。
>
> 剩余问题主要是：
>
> 1. 一处比较重要的 **Conclusion scope 矛盾**；
> 2. “confirmatory” 与新 H1 数值阈值之间还缺一点 protocol-timing 说明；
> 3. gradient-trained mapper 的 CI 信息压缩过度；
> 4. 若还愿意增加最后一个实验，最有价值的已经不是再造 mapper，而是**用 audit protocol 审计一篇公开声称 positive gain 的方法**。

也就是说，这篇论文现在已经可以停止“证明自己不是错的”，进入**最后收口阶段**。

---

# 二、这一版最重要的进步

### 1. H1 的统计定义终于变成了可复现标准

现在不再写模糊的：

> cosine = 1 within float tolerance

而是明确规定：

$$
\cos(z_{\rm identity},z_{\rm self})\ge 0.9998
$$

并报告：

$$
\min=0.99986,\qquad \mathrm{mean}=0.99998.
$$



Abstract 也正确改成：

> “Mechanics passes to a numerical tolerance”

而不是“exactly”。

这是正确修改。

---

### 2. H2 的统计框架现在基本完整

最新版把我上一轮指出的一个重要问题彻底修掉了。

现在明确拆成两个独立判断：

$$
\text{Replacement pass}
\iff CI_L>-\epsilon
$$

和

$$
\text{Direction}
=
\begin{cases}
\text{degraded},&CI_U<0\\
\text{gain},&CI_L>0\\
\text{neutral},&\text{otherwise}
\end{cases}
$$

而且明确承认：

> practical non-inferiority 与 statistically worse 可以同时成立。



Table 1 也已经拆成 `Repl.` 和 `Dir.` 两列。

这一修改非常好。

现在这套 audit gate 已经具备被别人复用的条件。

---

### 3. H3 的 claim 现在终于和实验能力匹配

现在定义已经从早期非常危险的：

> teacher cache 是否 intrinsically contains exploitable advantage

收缩成：

> **tested teacher-derived cache content 是否在 controlled intervention 下给 Student 带来 incremental benefit。**

而且明确说明：

> null result 只约束 tested extraction paths，不排除 nonlinear paths，也不是关于 teacher-cache intrinsic information content 的结论。 

Section 5.3 标题也已经改成：

> **No Teacher-Content Probe Establishes Gain**



这个版本我认为已经站得住。

---

### 4. Heo-style baseline 现在终于达到了“合理对照”的水平

最新版已经真正加入：

$$
\text{top-k}
+
\text{concatenation}
+
\text{de-RoPE}
+
1024\text{-token web-text calibration}
$$

结果：

$$
k=1:\;-0.279
$$

$$
k=3:\;-0.223
$$

$$
k=5:\;-0.254
$$

而且三个 CI 均低于 0。

同时 Appendix 没有伪装成 exact reproduction，而是明确承认：

* 本文 generated web-text；
* reference 是 FineWeb-Edu；
* 本文 100 sequences；
* reference 500 sequences。



现在这个对照已经足够诚实。

我不再认为“必须跑 500 FineWeb-Edu”是 acceptance blocker。

---

### 5. Related Work 的 positioning 明显变强

CacheBridge 和 Universal Context-Reuse Layer 已经加入，并且 Introduction 现在非常合理地指出：

> 文献中已经同时出现 near-native retention、above-receiver-baseline gain 和 severe degradation，因此需要统一 audit。



这是现在这篇论文最好的 framing。

它不再是：

> “别人说成功，我们说失败。”

而是：

> **“领域里存在互相矛盾的成功/失败证据，因此需要统一测量框架。”**

这会让 Reviewer 更容易接受 audit paper 的独立贡献。

---

# 三、当前最重要的问题：Conclusion 现在仍然有一个实质性 scope 矛盾

这是我认为投稿前**必须修掉**的一处。

Conclusion 开头写：

> “Under strict zero re-prefill, on the pairs and designs we tested, translating a teacher’s KV cache into a smaller student **does not replace the student’s own prefill**.”



但这其实不是整个实验集的事实。

因为 cross-architecture 里存在：

$$
\text{Llama-3.2-1B}:
0.000[-0.012,+0.012]
$$

以及 token-aligned：

$$
-0.003[-0.011,+0.005]
$$

和：

$$
\text{Gemma-2-2B aligned}:
-0.007[-0.018,+0.005].
$$

在你定义的：

$$
\epsilon=0.02
$$

reporting margin 下，这些配置实际上 **clear replacement margin**。

因此 Conclusion 当前第一句把：

> **primary strong-student result**

错误推广成：

> **all tested pairs**

了。

### 建议直接改成

> **“Under strict zero re-prefill, no tested translator establishes replacement for our primary strong-student pair. The only configurations that clear the reporting replacement margin occur on near-chance alternative students and show no measurable gain.”**

这才和全文完全一致。

这是当前我认为最重要的一处修改。

---

# 四、H1 新阈值还有一个 protocol-timing 问题

虽然：

$$
0.9998
$$

这个数值 gate 很合理，但我会从 Reviewer 角度问一句：

> **0.9998 是什么时候决定的？**

因为上一版是：

> equality to one within float tolerance

现在根据观测到的：

$$
\min=0.99986
$$

改成：

$$
\ge0.9998.
$$

而 Section 3.3 又称：

> identity gate 是 confirmatory，fixed before the runs。



如果 **0.9998 是看到 0.99986 后才选的**，那严格来说：

* H1 gate **形式**可以是 confirmatory；
* numerical tolerance 0.9998 则不是。

这和你之前对：

$$
\epsilon=0.02
$$

处理得如此透明形成了对比。

### 我建议 Appendix B 增加一句

如果 0.9998 是事后数值化的，就直接写：

> *The identity-gate form was fixed before evaluation; the numerical cosine tolerance 0.9998 was introduced during revision as a reproducible operationalization of the original “within numerical tolerance” criterion.*

这样完全没问题。

反而会提高可信度。

如果它确实事前就固定，那就给 commit/tag。

---

# 五、H1 还有一个值得补充但很小的问题：为什么不是 1？

Identity cache 理论上来自 Student 自己，因此 Reviewer 可能仍然会问：

> 为什么 minimum cosine 是 0.99986，而不是几乎 machine-identical？

目前 gold-probability difference 确实极小：

$$
+0.0001[-0.0006,+0.0009]
$$

所以不影响结果。

但作为一篇把 **Identity Control** 当核心方法论贡献的论文，建议 Appendix 增加：

* max absolute logit difference；
* 或 mean KL；
* 或说明误差来自 dtype / cache serialization / kernel ordering / injection representation。

尤其 cold-start path 已经做到 recorded precision 一致，这进一步说明有必要解释 identity-vs-native 的残余数值差。

不是拒稿点，但会让 H1 更完整。

---

# 六、gradient-trained mapper 的统计证据在 13 页压缩版里反而少了一点

Table 1 对：

* per-head MLP；
* joint MLP

只报告：

$$
-0.39\sim-0.26
$$

这种 training ranges，却直接把：

> Repl = fail
> Dir = degraded

写死。

问题是你现在没有在表中给每个 training 的 CI。

对于 `degraded`：

$$
CI_U<0
$$

才是判据。

所以读者无法单靠最新稿确认：

> 所有 5 次训练是否真的都满足 upper CI < 0。

上一些版本其实有：

> worst CI upper bound

这样的信息，现在因为压缩稿件被删掉了。

### 建议 Appendix B 恢复一个极小表格

例如：

| Family    | runs | point estimate range | largest CI upper |
| --------- | ---: | -------------------: | ---------------: |
| MLP c30   |    5 |            −.39…−.26 |            −.077 |
| MLP c200  |    5 |            −.26…−.23 |            −.107 |
| Joint MLP |    5 |            −.30…−.26 |                … |

不占多少版面，却让 Table 1 的 `degraded` 判定可以被直接审计。

---

# 七、普通 bootstrap 从以前的 10,000 降到了 1,000，我建议改回 10,000

现在 Section 3.1 写：

$$
10^3
$$

paired percentile bootstrap resamples，而 H3 family-wise 用：

$$
10^4.
$$



对于大幅负结果无所谓。

但你有一些正好靠近 gate 的结果：

$$
[-0.020,+0.002]
$$

$$
[-0.018,+0.005].
$$

这些恰恰用于：

$$
\epsilon=.02
$$

附近的 replacement 判断。

1000 bootstrap draws 时 2.5% percentile 实际只落在大约第 25 个 order statistic，Monte-Carlo quantile noise 不算特别小。

对于一篇核心卖点是：

> **rigorous audit protocol**

的论文，我会直接统一用：

$$
10^4
$$

甚至：

$$
5\times10^4
$$

普通 paired bootstrap。

计算成本相对于模型推理几乎可以忽略。

这不会改变大部分结论，但能防止 Reviewer 在最不值得的地方挑统计实现。

---

# 八、有一个明确的文字错误：“teachers are close to chance”

Section 6.4 写：

> “two near-parity rows whose **teachers are close to chance**”

对应：

* Llama-3.2-1B；
* Gemma-2-2B。



这里明显应该是：

> **students / receivers are close to chance**

Teacher 是 Qwen3-4B，不是这里接近 chance 的模型。

这必须改，属于明显事实性笔误。

---

# 九、Section 5.2 还有一句统计语言最好收一下

现在写：

> “Every strong-student translation **degrades** the student relative to its own prefill”

但最佳 affine：

$$
-0.138[-0.317,+0.029]
$$

按照你自己 Table 1 的 Direction 判据，是：

> **neutral**

而不是 statistically degraded。 

所以建议改成：

> **“Every strong-student translation has a negative point estimate relative to self-prefill.”**

然后：

> “all but the best configuration are statistically degraded.”

这样就和新统计框架完全一致。

---

# 十、H3 “training-free” 这个词还可以再准确一点

Related Work 和 Section 7 都仍然强调：

> training-free probe。

例如：

> “Report a training-free probe of teacher-derived content...”



严格说 probe 本身确实没有额外训练。

但是 teacher-derived content 来源仍然是：

* RAT；
* affine translator；

它们本身需要构造/拟合。

所以 Reviewer 可能问：

> 到底哪里 training-free？

我会推荐统一改成：

> **“no-additional-training probe”**

或：

> **“fixed-translator diagnostic probe”**

更精确。

不是大问题，但能避免“oracle”时代同类歧义重新出现。

---

# 十一、Figure 2 对最新 Heo concat baseline 的呈现还略显不完整

Table 1 现在已经有：

* top-k average；
* top-k concatenated web-text。



但 Figure 2 中的 Heo-style 三根柱仍然看起来只对应旧的 c=200 average 版本。

因为 concat baseline 是这一轮非常重要的新结果，建议：

* 要么把 concat 三根柱加进去；
* 要么 Figure 2 caption 明确写：

> *The long-sequence concatenation variants are reported in Table 1 and omitted here for readability.*

否则 Reviewer 看 Figure 2 时会误以为最接近 published design 的新结果没有进入 landscape。

---

# 十二、现在最值得增加的“最后一个实验”已经变了

我现在**不建议继续增加第九种、第十种 translator**。

如果作者还有一次比较完整的实验预算，我认为最高价值的是：

> **拿一个公开声称 above-receiver-baseline gain 的方法，直接跑完整 H1/H2/H3 audit。**

例如你们现在引用的 Universal Context-Reuse Layer，论文自己写它：

$$
27.59\%\rightarrow34.48\%
$$

超过 receiver native baseline。

如果能够得到代码并复现：

* H1 identity；
* H2 replacement；
* strict gain；
* teacher-content probe；

结果无论是什么都非常有价值。

如果 gain 能经 audit 保留：

> 证明 framework 不只是“制造 negative result”，也能够确认 genuine positive transfer。

如果 gain 消失：

> 则 audit framework 的必要性更强。

这是现在唯一一个我认为有机会把论文从：

> **7/10 Accept**

推向：

> **8/10 Strong Accept candidate**

的实验。

但不是当前稿被接受的必要条件。

---

# 十三、最新版本最强的科学发现没有变化

现在我会建议作者在 rebuttal / presentation 中重点抓住这两个结论，而不是“translation fails”。

### 结论 A

$$
\boxed{
\text{better reconstruction}
\not\Rightarrow
\text{better downstream behavior}
}
$$

concatenation 的 \(R^2\) 随 \(k\) 提高：

$$
R_V^2:
0.326\rightarrow0.419\rightarrow0.428
$$

但 task CHG：

$$
-0.279,\;-0.223,\;-0.254.
$$

Joint MLP 同样提高 \(R_V^2\) 到 0.481，却仍然失败。

这是非常漂亮的结果。

### 结论 B

$$
\boxed{
\text{near-native PPL}
\not\Rightarrow
\text{task preservation}
}
$$

23.7 PPL 对 21.2 native-suffix reference，却仍然相差 23 accuracy points。

这两个结论的外延远大于：

> “我们没找到一个好的 mapper”。

它们才是这篇文章最有长期价值的地方。

---

# 十四、我现在的评分

| 维度                       | main(8) |        main(9) |
| ------------------------ | ------: | -------------: |
| Novelty                  |     8.0 |        **8.0** |
| Technical quality        |     8.5 |        **8.5** |
| Experimental rigor       |     9.0 |        **9.0** |
| Statistical rigor        |     8.0 |        **8.5** |
| Related work             |     9.0 |        **9.0** |
| Claim–evidence alignment |     9.0 |        **9.0** |
| Reproducibility          |     8.5 |        **8.5** |
| Clarity                  |     8.5 |    **8.5–9.0** |
| Significance             |     8.5 |        **8.5** |
| **Overall**              |   **7** | **7 / Accept** |

---

# 十五、如果我是正式 Reviewer，我现在会写

> **Accept.**
>
> This revision has addressed essentially all of my major methodological concerns. The audit now operationalizes the identity control with an explicit numerical tolerance, cleanly separates practical replacement from statistical direction, scopes exploitability to tested teacher-derived content rather than intrinsic teacher-cache information, includes both averaged and concatenated variants of the closest published strict-zero-prefill design, and places its negative results in the context of recent work reporting both near-native retention and above-receiver-baseline gains.
>
> I find the strongest contribution to be methodological rather than the negative result itself. The paper shows that injection mechanics, receiver-cache replacement, and genuine teacher-derived gain are distinct claims that require distinct controls. Its empirical findings further show that neither improved KV reconstruction nor near-native perplexity is sufficient evidence of functional capability preservation.
>
> My remaining concerns are minor. The conclusion currently over-generalizes the primary-pair replacement result despite several near-chance cross-architecture configurations clearing the chosen reporting margin. The numerical identity tolerance should be tied explicitly to the protocol timeline, and the gradient-trained rows should expose enough interval information to justify their “degraded” labels. I would also prefer more bootstrap resamples for near-margin decisions.
>
> **Score: 7/10 – Accept.**

---

## 十六、投稿前最终修改清单

这一次我只建议再做下面这些，不建议继续扩大研究范围：

1. **必须修：** Conclusion 第一段不要说所有 tested pairs 都不能 replacement；限定到 **primary strong-student pair**。
2. **必须修：** “teachers are close to chance” → **students/receivers are close to chance**。
3. **建议修：** H1 的 `0.9998` 阈值说明是事前还是 revision-added numerical operationalization。
4. **建议修：** “Every strong-student translation degrades” → “has a negative point estimate; all but the best are statistically degraded.”
5. **建议修：** gradient-trained rows补 worst CI bound，支撑 `degraded` 标签。
6. **建议修：** ordinary bootstrap 从 \(10^3\) 提升到 \(10^4\)。
7. **小修：** `training-free probe` → `no-additional-training diagnostic`。
8. **小修：** Figure 2 注明 concat Heo rows 是否因版面没有绘制。

如果这八项清完，我会认为这已经是一个**相当稳定的投稿版本**。

最重要的是：这篇论文现在已经从早期的“试图证明 KV cache 不可转移”，走到了一个更成熟的位置——**定义什么证据才足以声称跨模型 cache translation 真正实现了 replacement 或 capability gain。**

这个贡献即使未来更强的 KV translator 成功了，也不会失效。

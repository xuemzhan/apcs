我把这版 16 页最新稿按 **ICLR / NeurIPS / ICML 主会正式审稿标准**重新完整审了一遍，包括正文、Figure 1–4、Table 1–3、Limitations 和 Appendix。相较第三稿，这一版已经发生了质变：弱学生 non-inferiority 的统计结论、PPL 口径、cross-architecture control、joint MLP、evidence tier、probe 的边界以及复现说明，都明显更成熟。

## 一、最新审稿结论

**当前建议：6/10，Weak Accept。Reviewer Confidence：4/5。**

如果碰到偏 empirical / measurement / representation-analysis 的 Reviewer，我认为已经有较大概率给 **6**；如果碰到特别重视“是否复现最强 existing baseline”和严格统计推断的 Reviewer，仍有可能给 **5**。

与上一稿相比，我愿意把评分从 Borderline 5 提升到 6，主要因为三个关键问题已经真正解决：

* 0.6B 不再被错误宣称为通过 non-inferiority，Abstract 已明确写成“not established as non-inferior”，Section 5.2 也改成了 **weak-student result is inconclusive**。
* 21.2 与 25.4 的 PPL 差异终于被解释清楚：identity 是 suffix-conditioned PPL，而 student self-prefill 表中是 full context + suffix 的 native loss，两者不是同一个量。
* 作者现在明确区分 confirmatory / frozen-gate extension / exploratory evidence，而不是把修改过程中追加的所有实验都包装成预注册验证。这个处理非常专业。

而且 Conclusion 已经非常克制：不再声称不存在 nonlinear path，而是明确说“no configuration we test”，并承认 nonlinear / jointly structured extraction path 尚未排除。

所以现在这篇论文的主要问题已经从“核心结论是否成立”，转变成：

> **论文主体成立，但仍有几个可能影响顶会审稿结果的实验覆盖和统计表述问题。**

---

# 二、我现在认为论文最强的贡献是什么

这版之后，我会把论文理解成一篇 **audit / measurement methodology paper**，而不是一篇 KV translator paper。

这个定位已经越来越成立。

首先，三层问题的拆分很清楚：

$$
H1:\text{pipeline mechanics}
$$

$$
H2:\text{replacement}
$$

$$
H3:\text{teacher-readable advantage}
$$

Figure 1 现在不仅把工作流画出来，还明确给出 fixed gate、pre-specified 与 revision-added axes 的标识，信息密度和逻辑完整性都比前稿好很多。

其次，**replacement 与 gain 分开**是这篇文章真正有传播价值的概念贡献。现在 H2 正确地定义为：

$$
CI_{\text{lower}}>-\epsilon,\qquad \epsilon=0.02
$$

而 gain 需要：

$$
CI_{\text{lower}}>0
$$

并且不再错误要求 point estimate 本身必须非负。 

第三，**PPL–capability decoupling** 仍然是我认为最漂亮的 empirical result：

$$
PPL=23.7
$$

已经接近 identity cache 的：

$$
21.2
$$

但 accuracy 仍然只有：

$$
0.267
$$

而 student 是：

$$
0.500
$$

这是一条非常清楚的结果：

> **Fluent cache ≠ capable cache.**



第四，新加入的 joint MLP 也显著增强了论文。它不是一个小型 per-head nonlinear mapper，而是把所有 aligned teacher layers 和所有 KV heads 在同一 position 拼起来，约 37M 参数，仍然没有接近 student self-prefill。

这个实验确实降低了“你只是 linear mapper 太弱”的质疑力度。

---

# 三、当前最重要的 Major Concern：仍然没有真正复现最强的 strict-zero-re-prefill baseline

这是我现在认为**最有可能阻止论文从 6/10 稳定进入 7/10 的问题**。

论文已经正确地引用了 Heo et al. 的 closed-form cross-model KV transfer，并指出他们报告了：

> 73–98% retention



Table 1 也明确指出该方法包含：

* top-k layer selection；
* de-RoPE keys。



问题是：

> **你们仍然没有把这个最直接、最接近、而且同样属于 zero-re-prefill 的方法真正放到自己的 audit protocol 里跑。**

这对于一篇标题和 Introduction 都强调“audit recent cache-translation claims”的论文，是当前最大的实验缺口。

现在有七类 mapper 很丰富，但 Reviewer 很可能直接问：

> Why did the audit test seven translator families but omit the strongest published strict-zero-reprefill baseline?

尤其 top-k cross-layer selection 和 de-RoPE 并不是无关紧要的小 trick。

它们针对的正是论文当前分析出的核心问题：

* layer correspondence；
* key geometry；
* positional encoding；
* source-layer information availability。

所以从投入产出比看，如果现在还能补**一组实验**，我认为最值得补的不是更多模型，而是：

$$
\boxed{\text{Heo-style top-k source layers + de-RoPE ridge}}
$$

并且用：

* 同一 Qwen3-4B→1.7B；
* 同一 fixed eval set；
* 同一 CHG metric；
* 同一 ε=0.02 replacement gate；
* 同一 identity control。

这会极大增强论文。

如果它仍然失败，你的 negative conclusion 会非常强。

如果它达到 replacement，论文也不会失败，反而会得到更有价值的发现：

> **失败来自 translator design，而不是 cache transfer 本身；audit framework 成功识别了边界条件。**

这甚至可能让论文更有影响力。

---

# 四、Section 2.2 对 Heo et al. 的最后一句仍然说得稍微过头

目前写：

> “Under the strict regime ... the gap is a property of the regime, not a contradiction of those results.”



这里我还是建议改。

因为 Heo 本身也是 prefill reuse / zero-target-prefill 类型工作。

所以你不能把双方差异概括为：

> strict regime 的 property。

更准确应该是：

> **Under our strict zero-reprefill protocol, evaluated model pairs, calibration regime, and translator families, the deficit persists; this does not contradict the positive retention reported under Heo et al.'s different layer-selection and key-alignment design.**

也就是说，不要把因果归到：

$$
\text{zero re-prefill}
$$

应该归到：

$$
\text{our tested regime}
$$

包括：

* pair；
* calibration；
* metric；
* layer selection；
* de-RoPE；
* mapper family。

这是一个小改动，但会避免熟悉 Heo 工作的 Reviewer 产生明显反感。

---

# 五、统计设计已经进步很大，但“全局 null”目前还不是严格意义上的统计结论

Section 3.3 是这版非常好的新增内容，但也留下一个细微却重要的问题。

作者写道：

> 搜索越多 configuration，本来越容易出现 false positive；但 119 个 translated configurations、多个 probes、多个 pairs 中仍没有 positive CI，因此 null 更有说服力。

作为**描述性论证**，我同意。

但如果想把它写成严格 confirmatory statistical claim：

> “there is no positive effect in the audited family”

还差一步。

因为：

$$
\text{failure to reject } H_0
$$

本身不是：

$$
\text{evidence that } H_0\text{ is true}.
$$

尤其 H3 的很多 probe 只有：

$$
n=30
$$

你真正需要的不是简单看：

> 有没有某一个 95% CI 完全高于 0，

而最好给一个：

### family-wise one-sided upper bound

例如对 confirmatory H3 configurations 做：

$$
T=\max_j \Delta_j
$$

然后通过 paired bootstrap / permutation 得到：

$$
U_{0.95}\left(\max_j\Delta_j\right)
$$

如果连这个 global upper bound 都很小，就可以非常漂亮地说：

> **Across the confirmatory probe family, any positive gain larger than X is excluded at 95% confidence.**

这会比：

> “none of the individual intervals excludes zero positively”

强很多。

---

# 六、Multiplicity 部分的逻辑需要再严谨一点

现在的写法大致是：

> 搜索越多 configuration，越容易出现 false positive，所以搜索这么多还没有 positive makes the null more informative.

方向上没错。

但如果这套 audit 要作为未来论文的“标准 protocol”，那么未来一旦出现：

> 某一个 probe CI > 0

就必须面对 multiple testing。

例如：

* 3 fraction probes；
* 3 thirds；
* 8 octants；
* multiple source translators；

如果逐个使用：

$$
\alpha=0.05
$$

那么“some configuration is positive”这个 H3 gate 的 family-wise false-positive rate 会膨胀。

因此我建议 Section 3.2 / 3.3 加一句：

> *For future positive H3 claims, the “some configuration” gate should be evaluated with a family-wise correction or a max-statistic permutation test. Because no positive interval occurs in our data, such correction cannot create a positive finding here.*

这样 audit framework 才真正完整。

---

# 七、Table 2 的 Gate 定义还有一个统计措辞错误

Table 2 caption 现在把：

> `n.e.`

定义成：

> “CI includes zero, so replacement is not established”



这和你自己的 non-inferiority framework 并不完全一致。

因为一个区间完全可以：

$$
[-0.012,+0.012]
$$

包含 0，但仍然：

$$
-0.012>-0.02
$$

因此通过 replacement gate。

事实上 Table 3 的 Llama-3.2-1B 就正是这样：

$$
+0.000[-0.012,+0.012]
$$

并被正确标成 **pass**。

所以 `n.e.` 不应该定义为：

> CI includes zero。

建议定义为：

> **n.e. = lower CI ≤ −ε and upper CI ≥ 0; neither non-inferiority nor significant degradation is established.**

而：

> **deg. = upper CI < 0**

> **pass = lower CI > −ε**

这样数学上才完全一致。

这个问题不改变任何实验结论，但正式 Reviewer 很容易看到。

---

# 八、H1 现在仍然有一个概念措辞问题

Figure 1 把 H1 描述为：

> “Is a foreign cache consumed without loss?”



但 identity control 实际做的是：

$$
C_S\rightarrow\text{student injection pipeline}\rightarrow S
$$

也就是把 **student 自己的 cache** 重新注入。

因此它严格证明的是：

> **student-space cache injection mechanics are correct.**

它并没有证明：

> 任意 foreign cache 被无损消费。

实际上 foreign translated cache 的：

* geometry；
* distribution；
* token alignment；
* layer correspondence

恰恰是 H2 的问题。

我建议 H1 改名为：

> **Injection mechanics**

问题改成：

> *Does the cache-injection pipeline preserve a valid student-space cache without loss?*

这样逻辑更精确。

同样 Section 5.1：

> “every subsequent gap is attributable to the translation itself”



最好稍微改成：

> “not attributable to the student-space injection mechanics.”

因为后续 gap 可能来自：

* mapper；
* layer alignment；
* token alignment；
* translated-cache distribution shift；

这些都属于 translation regime，但不一定单纯是映射函数。

---

# 九、PPL 问题基本解决了，但 Figure 4 仍有一个内部不一致

这版对 21.2 vs 25.4 的解释已经非常好：

> identity PPL = suffix conditioned on injected cache；
> self-prefill 25.4 = full context + suffix native loss。



Table 2 caption 也明确说：

> 这两个 PPL **not directly comparable**。

问题是 Figure 4 仍然把：

> student self-prefill / identity control

放在同一个 PPL 横轴中一起画。

这会让视觉读者自然理解：

$$
25.4
$$

和：

$$
21.2
$$

是同一个量。

最好有两个方案之一。

**更推荐：**

重新计算 native student 的 **suffix-only PPL**，使用与 injected-cache 完全相同的 suffix scoring protocol。

这样：

$$
PPL_{\text{student,suffix}}
\approx PPL_{\text{identity}}
$$

应该自然重合。

然后 Figure 4 全部使用可比较的 suffix PPL。

如果不重新计算，就把 self-prefill 的 25.4 point 从 Figure 4 删除，只保留 identity control 作为 native-suffix reference。

另外 Section 5.2 还有一句：

> joint mapper reconstructs a fluent cache “PPL 29.6 versus 25.4 for the student”



按照你自己的定义，这个比较也是不成立的。

应该写：

$$
29.6\quad vs.\quad21.2\text{ identity suffix PPL}
$$

不是 25.4。

这是我建议投稿前**一定修**的 consistency issue。

---

# 十、Abstract 中的 R² 结论还是略强

Abstract 现在说：

> keys \(R^2=0.81\)，values \(R^2=0.32\)，
> “consistent with a per-head information deficit rather than a limit of mapper capacity.”



这个措辞仍然偏强。

你真正测量的是：

> **linear held-out recoverability**

而不是：

$$
I(V_T;V_S)
$$

这样的 information-theoretic information content。

Section 6 的实验本身做得不错：

* 20 contexts fit；
* 10 held-out；
* per-(layer,head) R²；
* token positions pooled；
* affine key R² = .81；
* value R² = .32；
* ridge 得到类似结果。



但：

$$
R^2=0.32
$$

只能证明：

> **teacher per-head values do not linearly predict the corresponding student values particularly well under the tested map.**

不能严格推出：

> information 不在 teacher cache 中。

尤其 joint MLP 和 cross-attention 仍可能利用：

* multi-head；
* multi-layer；
* cross-token structure。

论文自己在 Limitations 已承认 large cross-attention translator 没有覆盖。

所以建议 Abstract 改成：

> **“consistent with incomplete per-head recoverability rather than simple calibration error.”**

或者：

> **“consistent with a structural mismatch in per-head value states.”**

避免使用 “information deficit”。

---

# 十一、Section 5.2 又重新出现了一个上一版已经改掉的过强结论

当前写：

> “The mapper has converged, and more data cannot close a gap that is set by the information content of the translated cache rather than by estimation variance.”



我建议再次改回上一版更好的措辞：

> **The affine mapper appears calibration-saturated within the tested budget; additional calibration examples in this range do not close the gap.**

你现在证明的是：

$$
10\rightarrow200
$$

样本下，固定 evaluation set，affine mapper 表现稳定。

你没有证明：

$$
N\rightarrow\infty
$$

仍然不可能改善。

更没有严格证明：

> gap 是 information content 导致。

joint MLP 失败增强了这个解释，但仍然不是 information-theoretic proof。

同一段的：

> “Nonlinearity does not help.”

也建议改成：

> **“Neither tested nonlinear family closes the gap.”**

虽然你现在的证据比以前强很多，但仍然没必要给 Reviewer 留 universal-claim 的攻击点。

---

# 十二、Joint MLP 很重要，但最好补一个训练/泛化诊断

37M joint MLP 是这版很有价值的新实验。

但 Reviewer 仍可能问：

> 失败到底是 representation 不可恢复，还是优化失败？

现在只看到 downstream：

$$
CHG\approx -0.26\sim-0.30
$$

和：

$$
PPL=29.6
$$



最好增加一小张 appendix table，报告：

$$
\text{train KV reconstruction loss}
$$

$$
\text{held-out KV reconstruction loss}
$$

或 R²。

最关键的是回答：

### 情况 A

如果 joint MLP：

$$
R^2_{\text{train}}\approx1
$$

但：

$$
R^2_{\text{test}}\ll1
$$

说明主要是：

> calibration/generalization limitation。

### 情况 B

如果 train R² 都很低：

说明：

> optimizer / capacity / architecture 本身没拟合好。

### 情况 C

如果 held-out reconstruction 很好，但 downstream 仍然差：

这是最强的结果：

> **cache-state reconstruction fidelity itself does not guarantee capability preservation.**

这会把论文推到一个更深的结论。

目前的 joint MLP 只证明：

> 一个较大的 position-wise MLP 也没有恢复 task quality。

这已经有价值，但还没完全回答为什么。

---

# 十三、最好明确把 Joint MLP 称作 “position-wise joint MLP”

当前 Joint MLP 虽然一次读：

* all layers；
* all heads；

但它仍然是：

> **at a position**



也就是说它并没有跨 token positions 建模。

所以它不等价于：

* cross-attention translator；
* sequence-level translator；
* context-aware translator。

建议在全文统一称：

> **position-wise joint MLP**

否则 “joint” 容易让 Reviewer误以为已经覆盖完整 structured nonlinear map。

这也和 Limitations 中：

> large-scale trained cross-attention translator 尚未覆盖

形成自然呼应。

---

# 十四、Section 6 最后仍然有一句需要软化

现在写：

> “the teacher’s advantage over the student sits in its parameters.”



而 Conclusion 已经非常谨慎地写：

> “evidence is consistent with an advantage that is weight-mediated”



我建议 Section 6 和 Conclusion 对齐。

改成：

> **“The observed teacher–student advantage is consistent with being mediated by parameter-dependent computation that raw KV transfer does not reproduce.”**

因为：

> cache 中是否完全没有可提取的 advantage

你没有证明。

而：

> tested translator 无法把它提取成 student-readable form

你证明得相当不错。

---

# 十五、H3 现在已经解释得很好，但正文还残留两处“bound”旧表述

Section 4.4 已经正确写：

> “diagnostic interventions rather than universal upper bounds”



Limitations 也正确承认：

> nonlinear or jointly structured path would not be detected。

但 Section 2.5 仍然写：

> “bounds exploitability with an oracle”



Section 7 又写：

> “oracle probes ... bounding exploitability without training anything.”



这两个地方应该彻底改掉。

统一成：

> **probe / constrain exploitability within controlled interventions**

否则一个认真读全文的 Reviewer 会发现内部逻辑冲突。

---

# 十六、H3 的 octant experiment 是很好的改进

这一版对 layer-window 的解释已经比上一稿严谨很多。

现在不再说：

> early/mid layers 是 answer routing 发生的位置。

而改成：

> 这是 sensitivity profile，可能也来自 off-manifold perturbation sensitivity。



这是正确的因果表述。

而且明确报告两个 top octant：

$$
+0.020,\quad +0.006
$$

但 CI 跨零，没有把不利于 narrative 的正点估计隐藏掉。

这一点从审稿角度会增加可信度。

---

# 十七、Cross-architecture extension 有价值，但不要把它当成主证据

Table 3 现在覆盖：

* Llama-3.2-1B；
* Gemma-2-2B；
* Llama-3.2-3B；
* Qwen2.5-1.5B；
* Gemma-3-1B；

还有 token-aligned variants。

这是明显增强。

但它仍存在三个结构性限制：

第一，Teacher 仍然都是 Qwen3，没有 non-Qwen teacher。

第二，不同 tokenizer 下 position-by-position cache matching 本身就不是自然对应。

第三，head-group mean pooling + rectangular mapper 是一个相当粗糙的 alignment。

作者现在已经很谨慎地把这些实验定位为：

> mechanics and gate-level replacement checks rather than fine-grained capability measurements。

这个定位是正确的。

所以我建议 Abstract 的：

> “five alternative students”

可以保留，但不要把 cross-family 作为“generality proven”的卖点。

---

# 十八、Token-aligned control 很有帮助，但最好补 closest-to-margin 两个模型

目前 token-aligned 只跑：

* Llama-3.2-3B；
* Gemma-3-1B。



而真正最关键的反而是：

* Llama-3.2-1B：通过 replacement gate；
* Gemma-2-2B：恰好卡在 margin。

这两行目前没有 token-aligned control。

论文自己也诚实承认这是 limitation。

如果还能补两组小实验，我认为这是第二优先级。

因为当前唯一的 replacement-pass case：

$$
+0.000[-0.012,+0.012]
$$

恰好来自一个 token alignment 尚未控制的 cross-tokenizer student。

如果 token-aligned 后结果变化，那么：

> “唯一 pass 的 translated configuration”

这个 headline 可能变化。

因此最好在投稿前补齐。

---

# 十九、“long-context”这个词我建议降级

现在所谓 long-context task 是：

$$
\approx1024\text{ tokens}
$$



到了 2026 年，1K token 很难被 Reviewer 视为真正 long-context。

尤其论文开头的 serving motivation 是：

> teacher prefills a long context once。

我建议直接改成：

> **1K-token retrieval stress test**

或者：

> **longer-context needle task**

如果资源允许，最好增加：

$$
4K,\ 8K
$$

至少一个 context length。

甚至只做一个：

$$
8K,\ n=30
$$

也比现在把 1K 叫 long-context 更有说服力。

---

# 二十、如果目标是顶会 Strong Accept，任务覆盖仍然偏窄

当前核心仍是：

* HellaSwag；
* ARC-Challenge；
* needle MCQA。

全部是 multiple-choice。

作者已经解释为什么 MCQA 适合：

> gold probability 可比较、split 容易控制。

这个理由成立。

但 reviewer 仍可能问：

> cache translation 对 free generation 是否也是同一结论？

如果资源允许，增加一个小的 open-ended benchmark：

* GSM8K exact match；
* TriviaQA short answer；
* NaturalQuestions short answer；
* 一个固定长度 generation KL / token agreement。

不需要很大。

它主要是为了证明：

> failure 不只是 answer-letter scoring 的 artifact。

这不是当前接受的必要条件，但会显著提升外部效度。

---

# 二十一、gold probability 指标本身需要再写清楚一点

论文反复使用：

$$
p_{\text{gold}}
$$

作为核心连续指标。

但正文需要明确它到底是：

1. gold option letter 的 raw LM token probability；
2. 还是四个 answer letters 之间重新归一化后的 probability；
3. 还是完整 option string likelihood。

这是非常重要的。

如果是 raw token probability，那么它同时受：

> probability mass 是否分配给 A/B/C/D token

影响。

如果是 four-choice normalized probability，则更加适合作为 MCQA confidence metric。

建议在 Section 3.1 给公式：

$$
p_{\text{gold}}
=
\frac{\exp z_{y}}
{\sum_{c\in\mathcal C}\exp z_c}
$$

如果确实是这样。

并明确：

* single-token answer letters；
* 是否带空格；
* tokenizer 是否一致；
* cross-family student 是否各自使用其本地 answer-token IDs。

这是 Reviewer 很容易追问的 reproducibility detail。

---

# 二十二、Bootstrap / permutation test 仍然描述不足

正文说：

> 95% bootstrap intervals and sign-flip permutation tests。

但主要结果里几乎只看到 CI，没有看到 permutation p-value。

建议 Appendix A 至少补：

* bootstrap resamples 数量；
* paired bootstrap 还是 unpaired；
* percentile / BCa；
* one-sided or two-sided；
* permutation 次数；
* sign-flip statistic；
* p-value 如何使用。

如果 permutation test 最终完全没进入任何 decision gate，可以考虑删掉正文的强调，否则 Reviewer 会问：

> Why mention permutation tests if none of the reported decisions uses them?

---

# 二十三、预先规定（pre-specified）最好给一个真正可审计的时间锚

Section 3.3 已经非常努力地区分：

* confirmatory；
* revision-added frozen-gate；
* exploratory。

这是加分项。

但由于论文自己承认早期 protocol 存在：

* confidence-based metric；
* in-sample calibration；
* cache mutation bug；
* small positive deltas。



Reviewer 很可能继续问：

> When exactly was the confirmatory protocol frozen?

建议 Appendix A 给：

> **protocol-freeze git commit / tag**

例如：

> `audit-v2.0`, commit XXXXX, dated YYYY-MM-DD

并明确：

> confirmatory evaluation examples were not inspected under the audit metric before this freeze.

这样“confirmatory”就真正可审计。

否则很容易被理解成：

> 在多轮实验以后重新定义了一套被称为 pre-specified 的标准。

---

# 二十四、119 configurations / 162 runs 不宜被作为“样本量”来暗示证据强度

Abstract 现在写：

> seven mapper families ... 162 recorded evaluation runs。

正文又写：

> across 119 translated configuration rows...

作为复现透明度，这是优点。

但这些 runs 高度相关：

* 相同 model pair；
* 相同 eval examples；
* 相似 mapper；
* hyperparameter variants。

所以不能让 Reviewer 感觉作者在用：

$$
N_{\text{runs}}=162
$$

替代：

$$
N_{\text{independent evidence}}
$$

我建议 Abstract 把“162 recorded runs”移到 reproducibility contribution，不要与：

> four pairs, five students

并列成 evidence breadth。

可以写：

> “with all 162 audit runs recorded for reproducibility.”

语气会更准确。

---

# 二十五、Sample-size calculation 的 “500 examples”要注明条件

现在写：

> 4B→0.6B 若要通过 ε=.02，约需要 500 samples。



这个数字大概率是根据当前：

$$
\hat\Delta=0.01,\quad SD\approx0.40
$$

估算 one-sided CI width 得出的。

但这不是标准意义上的：

> 80% power sample size。

建议写成：

> **“Conditioning on the observed mean and variance, approximately 500 samples would be required for the one-sided 95% CI lower bound to exceed −0.02.”**

不要单纯叫：

> required sample size。

同样：

> “8B→0.6B point estimate already sits at margin and cannot clear it at any sample size”

应该改为：

> **“If the point estimate remained at −0.023, increasing n alone would not make the CI satisfy the −0.02 non-inferiority margin.”**

因为未来更大的样本下 point estimate 本身当然可能变化。

---

# 二十六、Teacher full prefill 不应该叫 “upper bound”

Figure 1 / Setup 把：

$$
T(q|C_T)
$$

称作：

> capability upper bound。 

严格来说它不是数学意义上的 upper bound。

完全可能出现：

$$
Student+translator > Teacher
$$

或者某些样本 Student 本身比 Teacher 更好。

所以建议统一改成：

> **teacher full-prefill reference**

或者：

> **teacher capability reference**

这样不会引发不必要的理论争论。

---

# 二十七、Native-cache control 需要把 layer mapping 讲清楚

作者说：

> injecting teacher’s native (norm-rescaled) cache entries directly

会导致显著下降。

但是：

$$
36\text{ teacher layers}\rightarrow28\text{ student layers}
$$

即使 KV heads 和 head dim 一致，也仍然需要某种 layer correspondence。

因此它不是真正意义上的：

> untranslated cache。

至少还有一个：

$$
\ell_T\rightarrow\ell_S
$$

mapping。

最好明确：

* depth-ratio？
* nearest layer？
* average？
* interpolation？

因此把这一项称为：

> **native-value / no-channel-map control under the fixed layer correspondence**

会比 “untranslated teacher content” 更精确。

---

# 二十八、Figure 2 的一句话建议修改

现在 caption：

> “No family reaches zero”



但至少 affine c=30 的 CI 包含 0。

显然作者这里说的是：

> point estimate。

最好改成：

> **“All primary-pair point estimates are negative, and no configuration passes the replacement gate.”**

这恰好与你们真正的统计标准完全一致。

---

# 二十九、Table 2 的 gradient-trained mapper 行最好补 CI 信息

MLP / Joint MLP 行现在报告的是：

> `−0.39 to −0.26`

这实际上是多个 training 的 point-estimate range。

但 Table caption 又说 Gold CHG 列是：

> CHG [95% CI]。



形式上不一致。

建议改成，例如：

> `−0.39 ... −0.26 (all 5 CIs upper<0)`

或者放 Appendix：

| family            | runs | point-estimate range | worst CI upper bound |
| ----------------- | ---: | -------------------: | -------------------: |
| per-head MLP c=30 |    5 |                  ... |                  ... |
| joint MLP         |    4 |                  ... |                  ... |

这样 Reviewer 能确认 “deg.” 是如何判定的。

---

# 三十、Section 7 对 PPL 的结论可以更精确

现在：

> “a perplexity gain is not evidence of capability”

这是好的。

但 Section 5.5 又说：

> “perplexity cannot serve as evidence of translation quality.”



“translation quality”太宽泛。

PPL 明明可以证明：

> fluent continuation quality。

你真正证明的是：

> **PPL is not sufficient evidence of task/capability preservation.**

建议统一成这句话。

---

# 三十一、1K needle + text-channel baseline 都不应成为核心 narrative

Limitations 对 text-channel baseline 已经非常诚实：

> n=30；
> per-sample scores 没保存；
> no CI；
> secondary baseline。



很好。

因此 Section 7 中：

> “honest fallback is the text channel”

最好也不要讲得太强。

当前只能说：

> **in this small secondary comparison, text compression loses less than the tested cache translation.**

而不是 deployment recommendation。

---

# 三十二、Appendix C 的 replay diagnostic 要更谨慎

Appendix C：

> 在 translated cache 上再让 student 重读 context，结果依然差，因此 naive replay 不足以恢复。

这里 Reviewer 可能会质疑：

> translated cache 已经代表同一 context，再把 context appended/re-read 一次是否本身就是一个不自然的 state？

如果位置 index 延续，实际上可能形成：

$$
[\text{translated context}][\text{same context again}]
$$

这不是普通 self-prefill。

所以不要让这个实验承担太多论证。

当前 Limitations 已经说：

> 不能代表 MoT full pipeline。

很好。

我建议 Appendix C 标题明确：

> **Append-on-translated-cache replay diagnostic**

并避免：

> “recovery requires X”

这种因果式结论。

只说：

> this naive replay variant does not recover performance。

---

# 三十三、论文现在最值得继续强化的理论解释其实是 K/V asymmetry

Appendix 里的 channel asymmetry 很有意思：

* value-only accuracy：0.367；
* key-only：0.300；
* both：0.200；

但 PPL 排序反过来。

这和：

* key R² = .81；
* value R² = .32；

组合起来出现一个非常有趣的现象：

> **Keys are easier to reconstruct geometrically, but errors in keys may be more damaging because they alter routing; values are harder to reconstruct but partially useful content survives.**

这比“teacher capability lives in weights”更具体、更可验证。

我甚至建议作者考虑把 Section 6 的主解释从：

> weights vs cache

稍微转向：

> **recoverability × sensitivity asymmetry**

即：

$$
\text{structural recoverability}
\times
\text{functional sensitivity}
$$

Key：

$$
R^2\text{ high, but routing sensitive}
$$

Value：

$$
R^2\text{ lower, but value-only transfer performs best}
$$

这是很有意思的 mechanistic story。

如果后续还有精力，可以做：

$$
\Delta\text{attention map}
$$

或：

$$
\text{attention-target agreement}
$$

看看 mapped key 是否真的导致 attention misrouting。

这可能成为论文里比“weight-mediated advantage”更原创、更扎实的机制解释。

---

# 三十四、我对当前稿件的评分

| 维度                       |   上一稿 |                 最新稿 | 最新评价                                                    |
| ------------------------ | ----: | ------------------: | ------------------------------------------------------- |
| Novelty                  |   7.5 |             **8.0** | audit framing + evidence tiers 更完整                      |
| Technical Quality        |   7.0 |             **7.5** | joint MLP、cross-family、octant probe 增强                  |
| Experimental Rigor       |   7.5 |             **8.0** | 体系已经相当完整                                                |
| Statistical Rigor        |   5.0 |             **7.0** | non-inferiority 已修正，但 global null/multiplicity 还能加强     |
| Related Work             |   8.0 |             **8.0** | 主要相关工作覆盖较好，但最强 baseline 未复现                             |
| Claim–Evidence Alignment |   7.0 |             **8.0** | Abstract/Conclusion 已明显克制                               |
| Reproducibility          |   8.5 |             **9.0** | run metadata、independent reproduction、module paths 都很好  |
| Clarity                  |   8.0 |             **8.5** | Figure 1 和 evidence-tier 结构清楚                           |
| Significance             |   7.5 |             **8.0** | 有成为 cache-transfer evaluation protocol 的潜力              |
| **Overall**              | **5** | **6 / Weak Accept** | 已跨过我个人接受线                                               |

---

# 三十五、如果我是正式 Reviewer，我现在会给出这样的最终意见

> **Weak Accept.**
>
> This paper presents a careful audit of cross-model KV-cache translation under strict zero re-prefill. The work is substantially stronger than a conventional negative-result study because it explicitly separates injection mechanics, replacement-level fidelity, and teacher-advantage exploitability; introduces an identity control; uses non-inferiority gates rather than equating non-significance with equivalence; distinguishes confirmatory, revision-added, and exploratory evidence; and demonstrates a notable decoupling between near-native suffix perplexity and downstream task accuracy.
>
> The empirical evidence is broad within the authors' chosen regime: seven mapper families, multiple model scales, nonlinear and joint translators, cross-architecture stress tests, token-alignment controls, layer-window and octant probes, and independent reruns. Importantly, the paper now scopes its negative result appropriately rather than claiming an impossibility theorem.
>
> My main remaining concern is that the audit still does not directly re-implement the strongest closely related strict-zero-prefill baseline, particularly the top-k cross-layer, de-RoPE closed-form method that reports substantially higher retention. Because this method targets precisely the layer- and key-alignment issues highlighted by the paper, evaluating it under the proposed audit protocol would materially strengthen the central empirical conclusion.
>
> I also recommend tightening the global statistical interpretation of the H3 probe family, fixing several remaining inconsistencies in the PPL presentation, and softening a few residual information-theoretic or universal statements. These concerns affect the strength and presentation of the conclusions more than the core validity of the audit.
>
> **Score: 6/10 — Weak Accept.**

---

# 三十六、投稿前，我建议按这个优先级处理

1. **最高优先级：补 Heo-style top-k + de-RoPE baseline。** 如果只能再做一组实验，就做这个。

2. **修 Figure 4 / PPL 一致性。** 29.6 应和 identity suffix PPL 21.2 比；最好重新算 student suffix-only PPL，全部统一口径。

3. **把 global H3 inference 做严谨。** 最好给 confirmatory probe family 一个 max-statistic / simultaneous one-sided upper bound，并说明未来 positive gate 需要 multiplicity correction。

4. **删掉最后几处过强表述。** 重点是：

   * “bounds exploitability”；
   * “more data cannot close the gap because information content”；
   * “Nonlinearity does not help”；
   * “teacher advantage sits in parameters”；
   * “gap is a property of the regime”。

5. **修几个统计/表格细节。** Table 2 的 `n.e.` 定义、500-sample 的条件性解释、Figure 1 的 H1 gate、MLP range 与 CI 表头一致性。

6. **有余力再补两个提升项。** 一是给 Llama-3.2-1B / Gemma-2-2B 做 token-aligned control；二是把所谓 long-context 从 1K 扩到至少 4K/8K，或改名为 1K retrieval stress test。

我现在的判断已经和前几轮不同：**这篇论文不再需要继续“证明自己不是错的”，主体已经站住了。接下来最重要的是用最直接的强 baseline 和更严格的统计表达，把 Reviewer 最后几个攻击面封住。**

如果把 **Heo-style baseline + PPL一致性 + H3全局统计** 三件事处理掉，我会把它从目前的 **6/10 Weak Accept** 提升到 **7/10 Accept 候选**。

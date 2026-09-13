我把这版 **19 页最新稿**重新按 ICLR / NeurIPS / ICML 主会标准审了一遍，并重点检查了上一轮要求补的几项：Heo-style baseline、H3 family-wise 统计、PPL 统一口径、margin-row token alignment、Joint MLP reconstruction、4K retrieval、cold-start persistence、统计协议说明和 Appendix。

这版已经明显进入“**可以投稿**”的状态。上一轮我认为阻止论文从 6 分进入 7 分的几个主要实验缺口，大部分已经补上。当前我的总体判断是：

> **当前版本：6/10，Weak Accept，而且比上一稿更稳。**
> **修改下面两个关键统计表述问题后，我会倾向 7/10 Accept。**
>
> 但我这次发现了一个新的、而且必须在投稿前修掉的内部矛盾：**正文反复把 \(\epsilon=0.02\) 称为 pre-specified / confirmatory，但 Appendix 明确记录它是在看到一部分实验结果之后才确定的。** 这个问题比继续补实验更重要。

---

# 一、这版相比上一版，确实解决了绝大多数主要问题

首先，上一轮最重要的缺口——**没有直接实现 Heo et al. 的 strict-zero-reprefill baseline**——已经补上了。

现在作者实现：

$$
\text{top-k cross-layer selection}
+
\text{de-RoPE keys}
+
\text{ridge}
$$

并在 \(k=1,3,5\) 上得到：

$$
-0.273,\quad -0.308,\quad -0.291
$$

对应 CI 全部在 0 以下。也就是说，在本文 Qwen3-4B→1.7B、固定 audit protocol 下，这一最接近 published strict-zero-reprefill 方法的实现同样没有接近 self-prefill。

这是非常重要的补强。

第二，Joint MLP 的证据现在真正有解释价值了。它不是只报告 downstream failure，而是进一步测了 reconstruction：

$$
R^2_K:
0.81\rightarrow0.854
$$

$$
R^2_V:
0.32\rightarrow0.481
$$

但 task CHG 反而比 affine 更差，大约：

$$
-0.27\quad vs.\quad -0.14
$$

这比以前单纯说“MLP 也失败”强很多，因为它提供了一个非常清晰的反例：

> **更好的 KV reconstruction fidelity 并不自动转化成更好的 task capability preservation。**



第三，H3 的统计处理已经明显升级。现在不只是逐个看 95% CI，而是对 6 个 confirmatory probes 做了 max-statistic bootstrap，得到最大 positive effect 的 95% upper bound：

$$
+0.009
$$

也就是在这个 confirmatory probe family 中，可以排除大约超过 **1 个 gold-probability point** 的平均正收益。作者同时诚实报告，如果扩到全部 22 个、包括 revision-added configurations，upper bound 会放宽到：

$$
+0.075
$$

这个呈现非常好。

第四，PPL 的问题基本解决了。Figure 4 现在全部使用 suffix-scoring protocol，不再把 full-context PPL=25.4 和 suffix-conditioned PPL 混画；caption 也明确说明 student full-context native loss 不进入图。

第五，margin-row 的 tokenization confound 也补掉了。Llama-3.2-1B 和 Gemma-2-2B 都补了 token-aligned result，依旧属于 near-chance、无 measurable gain 的 replacement-pass configuration。

所以实验层面，我认为现在已经不是“缺关键 baseline”的状态了。

---

# 二、现在最严重的问题：\(\epsilon=0.02\) 到底是不是 pre-specified？

这是这一版**投稿前必须修掉的第一问题**。

正文 Section 3.3 现在写：

> “Confirmatory claims were fixed before the runs: ... the replacement and gain gates at \(\epsilon=0.02\) on the primary pair...”



Abstract 也写：

> “pre-specified replacement margin (\(\epsilon=0.02\))”



正文多个地方继续使用：

> “pre-specified margin”



但是 Appendix A 非常明确地记录：

> gate 的**形式**是在 2026-08-29 固定的；
> numeric margin \(\epsilon=0.02\) 是到 **2026-09-12** 才确定；
> 而 calibration ladder 已经在 8 月 30–31 日跑过；
> 因此 \(\epsilon=0.02\) “**is therefore a reporting choice, not a pre-registered constant**”。



这是一个直接的内部矛盾。

而且它不是普通措辞问题。

如果 Reviewer 读到 Appendix，很可能会马上指出：

$$
\epsilon=0.02
$$

是在部分结果已知之后选择的，那么不能在主文中继续称：

> pre-specified / confirmatory margin。

### 我建议不要试图解释，而是直接统一改正

正确的 evidence hierarchy 应该是：

**真正预先固定的：**

$$
\text{non-inferiority gate form}
$$

以及：

$$
\text{gain gate}: CI_{\text{lower}}>0
$$

**后续选择的 practical reporting margin：**

$$
\epsilon=0.02
$$

因此正文建议改成：

> “We use \(\epsilon=0.02\) as a practical reporting margin, selected after the gate form was frozen; replacement results using this numeric margin are therefore sensitivity/reporting analyses rather than preregistered confirmatory tests.”

然后 Section 3.3 里的：

> “replacement and gain gates at ε=0.02 were fixed before the runs”

必须删除。

---

# 三、最好的处理方式：把 replacement 与 gain 的证据级别拆开

这一点其实不会伤害论文。

因为最重要的 central negative：

$$
\text{no tested configuration has a positive CI lower bound}>0
$$

完全**不依赖** \(\epsilon\)。

Appendix 已经正确指出：

> gain gate 与 ε 无关，而且没有 configuration 通过。

所以论文真正的主结论仍然很稳。

可以写：

> **Confirmatory gain result:** no tested primary configuration establishes a teacher-derived gain.

然后 replacement：

> **Secondary non-inferiority analysis:** using a practical reporting margin \(\epsilon=0.02\), only near-chance cross-architecture students clear the margin.

甚至可以附一个简单 sensitivity table：

| Margin            | 哪些 configuration 通过 replacement |
| ----------------- | ------------------------------- |
| \(\epsilon=0.01\) | ...                             |
| \(\epsilon=0.02\) | ...                             |
| \(\epsilon=0.05\) | ...                             |

这样 Reviewer 反而会觉得作者非常透明。

如果作者非常希望保住“\(\epsilon=.02\) confirmatory”这个表述，唯一干净的方法是：

> **在确定 ε=.02 后，再启用一个此前完全未查看过的新 hold-out evaluation set。**

然后把这个新 set 作为 confirmatory non-inferiority evaluation。

否则不要使用 “pre-specified ε=.02”。

---

# 四、H2 还有一个更深的统计逻辑问题：“no pass”并不等于“H2 false”

当前标题仍然是：

> **H2 Fails for the Strong Student**



但是仔细看 Table 2，最佳 affine c=30：

$$
CHG=-0.138,\quad
CI=[-0.317,+0.029]
$$

它没有通过 replacement gate，但也没有证明 degradation。

对于一个 non-inferiority test：

$$
H_0:\Delta\le-\epsilon
$$

$$
H_1:\Delta>-\epsilon
$$

没有拒绝 \(H_0\)，严格来说只能说明：

> **non-inferiority is not established**

而不是：

> **inferiority is established。**

这一点对很多 mapper 行当然问题不大，因为 CI 完全低于零，甚至明显 degradation。

但对于整个 existential H2：

$$
\exists g\in\mathcal G:\Delta_g>-\epsilon
$$

没有任何 configuration 成功“证明存在”，并不等价于证明：

$$
\forall g:\Delta_g\le-\epsilon.
$$

因此我建议把 Section 5.2 标题改成：

> **No Strong-Student Mapper Establishes Replacement; the Weak-Student Result Is Inconclusive**

这比：

> H2 Fails

更加准确。

Conclusion 里的：

> “The tested maps do not reach replacement”

最好也改成：

> **“No tested strong-student map establishes replacement under the audit gate.”**

这是一个细微但非常专业的区别。

H3 现在其实已经处理得更好——作者说：

> “no probe configuration delivers an interval-supported advantage”

而不是：

> “H3 is false”。

H2 应该采用完全相同的标准。

---

# 五、Heo-style baseline 是非常重要的新增，但现在还需要补“实现保真度”说明

加入 Heo-style baseline 后，我上一轮最大的 objection 基本消失了。

但是 Reviewer 下一步很可能问：

> 这个 Heo-style implementation 到底多大程度上等价于 [10]？

目前正文只说：

> per-target-layer top-k source selection by held-out ridge reconstruction score + de-RoPE keys, \(k\in\{1,3,5\}\), c=200。



我建议 Appendix 再补一张极小的 implementation-difference table，明确：

> original [10] vs this audit implementation

至少解释：

* top-k 是 **per layer** 还是 per-head-per-layer；
* K/V 是否独立选 layer；
* layer selection 的 validation samples 从哪里来；
* selection 以后是否重新用全部 c=200 refit；
* ridge λ 如何选；
* candidate source layers 是全部 teacher layers 还是局部窗口；
* 原论文 calibration data volume 与本文 c=200 task-context calibration 的差异。

这里尤其有一个值得 Reviewer 追问的问题：

> **“held-out ridge reconstruction score”使用的 held-out data 从哪来？**

如果 c=200 又要拿其中一部分做 layer selection，那么它实际上可能比普通 c=200 affine/ridge 使用更少 fitting data。

最公平的流程应该明确：

$$
\text{selection split}
\rightarrow
\text{choose top-k}
\rightarrow
\text{refit on full calibration set}
\rightarrow
\text{test}
$$

如果确实如此，就直接写出来。

---

# 六、Heo baseline 的 PPL 爆炸本身值得解释一下

结果非常显眼：

$$
k=1:PPL=78.5
$$

$$
k=3:PPL=2629.6
$$

$$
k=5:PPL=68966.1
$$



这会让熟悉 [10] 的 Reviewer 问：

> 为什么 top-k 越大反而严重失稳？是该方法本身在这个 pair 上失败，还是 implementation / conditioning / regularization 出现问题？

我不认为必须再跑大量实验。

但至少应该报告 Heo-style mapper 的：

$$
R^2_K,\quad R^2_V
$$

或者 calibration / held-out reconstruction error。

如果随着 k 增大 reconstruction 也变坏，那么：

> 这是模型 pair / geometry 上的问题。

如果 reconstruction 变好但 PPL 爆炸，那反而是一个非常有趣的现象：

> **更好的 local reconstruction 可能破坏 attention geometry。**

这会和你现在的 Joint MLP 结论形成呼应。

---

# 七、Conclusion 目前有一个非常明显的数量错误

Section 5.2 现在明确说：

> **Three translated configurations clear the replacement margin**

并进一步说明：

* Llama-3.2-1B unaligned；
* Llama-3.2-1B aligned；
* Gemma-2-2B aligned。



Figure 1 也正确写成：

> two near-chance students, three aligned/unaligned configurations。

但 Conclusion 仍然写：

> “**the one configuration** that clears our replacement margin...”



这必须改。

建议写：

> **“the only configurations that clear the chosen replacement margin correspond to two near-chance students and show no measurable gain.”**

不要再数“一个/三个”，这样更加稳定。

---

# 八、Figure 1 还残留了一个上一轮已经改掉的旧表述

正文 Setup 已经正确把 Teacher full prefill 改成：

> **teacher full-prefill reference**



但是 Figure 1 caption 仍然说：

> “teacher’s full prefill as the **capability upper bound**”



这个应当统一改成：

> **teacher full-prefill reference**

因为 Teacher 不是数学意义上的 upper bound。

理论上某个 Student+translator 完全可能：

$$
P_{\text{student+translator}}>
P_{\text{teacher}}
$$

所以不要给 Reviewer 一个没有必要的攻击点。

---

# 九、Figure 1 的 dagger 标记也受到 ε 时间线问题影响

Figure 1 H2 gate 写：

$$
CI_{lower}>-\epsilon,\quad\epsilon=.02
$$

并打了表示“fixed before runs”的 † 标记。

但 Appendix 明确说 numerical ε=.02 后来才定。

所以 Figure 1 也必须修。

我建议把标记拆开：

> † = gate **form** fixed before audit runs
> § = numerical margin selected later as reporting tolerance

否则一个认真 Reviewer 会认为 authors 在主文与 appendix 中对 preregistration history 给出不同说法。

---

# 十、Introduction 的 H1 还残留旧定义

Introduction 仍然说：

> H1 asks whether “**a foreign cache is consumed without loss**”



但 Figure 1 已经正确改成：

> **Does the cache-injection pipeline preserve a valid student-space cache without loss?**



后者才是 identity control 真正证明的事情。

因为 identity control 输入的是：

$$
C_S
$$

不是：

$$
C_T
$$

所以 H1 严格验证的是：

> **student-space injection mechanics**

而不是：

> arbitrary foreign cache mechanics。

建议把 Introduction、Figure、Section 5.1 全部统一成：

> **H1: injection mechanics**

这会让三层逻辑更加漂亮：

$$
H1:\text{student-space injection}
$$

$$
H2:\text{translation/replacement}
$$

$$
H3:\text{teacher-derived advantage}
$$

---

# 十一、H3 的 family-wise bound 很好，但证据级别还应说明

现在 6 个 confirmatory probe 的 max-bootstrap upper bound：

$$
+0.009
$$

确实是一项重要增强。

但这个 **max-statistic analysis 本身** 看起来是在 revision 中新增的，而不是 8 月 29 日提前固定的。

因此：

> underlying configurations 是 confirmatory；

不等于：

> family-wise max statistic analysis 也是 confirmatory。

建议加一句：

> *The family-wise max-bootstrap bound is a revision-added multiplicity analysis over the pre-specified confirmatory probe family.*

甚至标为：

> **frozen-data / post-hoc statistical consolidation**

都可以。

这样证据层级完全透明。

另外 Appendix 现在只说：

> max-statistic bootstrap uses same resampling scheme。

如果要称“family-wise upper bound”，我建议再写一行算法：

$$
T^{*(b)}
=
\max_j \bar\Delta_j^{*(b)}
$$

并说明所有 configuration 使用**相同 bootstrap indices**，保留 configuration 间 sample-level correlation。

这样非常清楚。

---

# 十二、Joint MLP 的 reconstruction diagnostic 已经很好，但可以再利用一步

这一版出现了我认为整篇论文很值得发展的 mechanistic result：

Affine：

$$
R^2_K=0.81,\quad R^2_V=0.32
$$

Joint MLP：

$$
R^2_K=0.854,\quad R^2_V=0.481
$$

但 Joint MLP task CHG 更差。

这说明真正值得强调的，不再只是：

> “KV information is missing”

而是：

> **state reconstruction and functional preservation are misaligned objectives.**

我认为这是比“weight-mediated advantage”更加扎实、更直接被实验支持的结论。

甚至可以把 Section 6 中的一部分 narrative 从：

> Why Translation Fails

转成：

> **Why Reconstruction Is Not Enough**

或者至少提出：

$$
\text{representation similarity}
\not\Rightarrow
\text{functional equivalence}
$$

这与论文已有的：

$$
PPL\not\Rightarrow accuracy
$$

形成非常漂亮的两层 decoupling：

$$
\boxed{
\text{KV reconstruction}
\not\Rightarrow
\text{task preservation}
}
$$

$$
\boxed{
\text{fluency}
\not\Rightarrow
\text{task preservation}
}
$$

这是当前论文最有长期价值的 insight 之一。

---

# 十三、Section 6 里关于 weights 的说法现在基本可接受，但仍建议再弱一点

当前已经从以前绝对化的：

> capability lives in weights

改成：

> “teacher’s advantage ... is **consistent with being mediated** by parameter-dependent computation”

这是明显进步。

不过后面又写：

> “the capability gap ... is a gap in these parameters, **which a cache does not transport**.”

这句话仍略绝对。

最好改成：

> **“which raw KV transfer does not explicitly transport.”**

因为 teacher KV 当然是这些参数计算出的 context-dependent state，其中可能隐含部分 capability signal。

你真正证明的是：

> 当前 translators 没能把这些信号变成 Student 可利用的状态。

不是：

> cache 中完全没有 parameter-mediated information。

---

# 十四、MoT 的机制解释还略有推断成分

Section 6 说：

> MoT’s correction loss is “exactly the upper-layer correction that strict zero re-prefill denies.”



这个“exactly”还是偏强。

Appendix C 已经非常谨慎地说：

> naive append-on-translated-cache replay 不能代表 MoT full pipeline。

正文建议与 Appendix 保持同样谨慎：

> **“MoT provides a target-side correction pathway that strict zero-reprefill explicitly disallows.”**

不要进一步解释成具体“upper-layer correction”。

---

# 十五、4K retrieval 的处理现在是正确的

上一轮我建议不要把 1K 称作 long-context。

现在作者已经明确改成：

> **1K-token retrieval stress test**

并补了约 4.8K-token run。

更重要的是，4K：

$$
CHG=-0.035,
\quad
CI=[-0.380,+0.313],
\quad n=20
$$

作者没有强行说成功或失败，而明确说：

> establishes neither degradation nor parity。



这是非常好的统计克制。

这一部分我不建议再继续追加 8K，除非投稿 venue 特别看重 long-context。

当前 4K 实验已经完成它应该完成的功能：

> 说明论文没有拿 n=20 的噪声结果硬讲故事。

---

# 十六、Gold probability 的定义现在也补得很好

现在终于明确：

$$
p_{\text{gold}}
=
\frac{e^{z_{c^\star}}}
{\sum_{c\in\{A,B,C,D\}}e^{z_c}}
$$

并说明每个模型使用自己的 answer-letter token ID。

这里我只再建议补一个细节：

> ARC-Challenge 是否只保留四选项样本？

因为指标明确只对：

$$
\{A,B,C,D\}
$$

归一化。

如果所有使用样本都是四选一，请直接写：

> *We retain only four-choice examples.*

如果存在非四选项样本，则必须说明如何处理。

这是一个很小但 Reviewer 可能会问的 reproducibility point。

---

# 十七、Bootstrap / permutation 统计说明现在基本够用了

Appendix 已经明确：

* paired percentile bootstrap；
* 10,000 resamples；
* same indices；
* sign-flip permutation 1,000 次；
* \(p=(\#extreme+1)/(n_{\rm perm}+1)\)；
* gates 完全由 CI 决定；
* p-value 不参与 decision。



这已经足够复现。

我唯一会问：

> 既然没有任何 decision 使用 permutation p-value，是否还需要在 Abstract/正文把 permutation test 作为主要 audit instrument 强调？

不是问题，只是可以精简。

---

# 十八、Figure 2 现在已经比较完整

Heo-style baseline 已进入 Figure 2，而且 caption 改成：

> **All primary-pair point estimates are negative and no configuration passes the replacement gate**

这比以前：

> “No family reaches zero”

严谨很多。

这一版 Figure 2 我认为基本可以保留。

---

# 十九、还有一个提交层面的严重问题：当前稿已经不是匿名稿

当前 PDF 首页直接包含：

* 作者姓名；
* 邮箱；
* CETC affiliation。



如果目标仍然是 ICLR / NeurIPS / ICML 这类双盲审稿 venue，**这一版不能直接提交**。

需要去掉：

> 姓名、单位、可识别邮箱、公开可反查身份的 repository 信息。

commit hash 本身未必有问题，但如果 repository 是公开实名仓库，也可能 deanonymize。

如果这是 arXiv / camera-ready 版本，则无所谓。

---

# 二十、当前论文的真正 story 已经非常清楚

我现在不会再建议作者继续围绕：

> “KV translation fundamentally impossible”

讲故事。

这篇文章现在最合理、也最有说服力的定位是：

> **A rigorous audit protocol for separating cache injection mechanics, receiver replacement, and genuine teacher-derived capability gain.**

核心 empirical findings 可以压缩成三句：

$$
\textbf{Identity works.}
$$

Student-space cache injection 本身不是问题。

$$
\textbf{Reconstruction does not imply functional preservation.}
$$

更高 KV \(R^2\) 不一定带来更高 task quality。

$$
\textbf{Fluency does not imply capability.}
$$

接近 native 的 PPL 不代表 downstream capability 被恢复。

我认为这三个点已经足够支撑一篇很好的论文。

---

# 二十一、我的最新评分

| 维度                       |   上一稿 |                 当前稿 | 评价                                           |
| ------------------------ | ----: | ------------------: | -------------------------------------------- |
| Novelty                  |   8.0 |             **8.0** | Audit framing 已经稳定                           |
| Technical Quality        |   7.5 |             **8.0** | Heo baseline + Joint MLP reconstruction 明显增强 |
| Experimental Rigor       |   8.0 |             **8.5** | 目前已相当完整                                      |
| Statistical Rigor        |   7.0 |             **6.5** | family-wise 分析增强，但 ε 时间线矛盾必须修                |
| Related Work             |   8.0 |             **8.5** | 最直接 baseline 已进入同协议测试                        |
| Claim–Evidence Alignment |   8.0 |             **8.0** | 总体很好，H2 wording/ε 仍需收口                       |
| Reproducibility          |   9.0 |             **9.0** | commit、run、seed、CI、module path 都非常完整         |
| Clarity                  |   8.5 |             **8.5** | 结构清晰，少数前后版本措辞残留                              |
| Significance             |   8.0 |         **8.0–8.5** | 有潜力成为领域 evaluation checklist                 |
| **Overall**              | **6** | **6 / Weak Accept** | 修正统计时间线后我会倾向 **7/10**                        |

这里统计严谨性反而从 7.0 临时降到 6.5，不是因为方法变差，而是因为 Appendix 现在把实验时间线披露得更清楚以后，暴露出了正文“pre-specified ε=.02”的不准确表述。

这反而是好事：现在这个问题非常容易修。

---

# 二十二、如果我是正式 Reviewer，现在会给出的评语

> **Weak Accept, close to Accept.**
>
> The revision substantially strengthens the empirical case. In particular, the authors now evaluate a Heo-style top-k/de-RoPE strict-zero-reprefill baseline under the audit protocol, add token-aligned controls for the previously near-margin cross-architecture cases, report a joint nonlinear mapper with held-out reconstruction diagnostics, unify the perplexity protocol, and introduce a family-wise upper bound over the confirmatory probe family. These additions address most of my prior concerns.
>
> The most interesting new result is that improved KV reconstruction does not translate into improved downstream performance: the position-wise joint MLP improves held-out key/value reconstruction yet performs worse than the simpler affine map. Together with the near-native-perplexity failure, this gives the paper a stronger message than a generic negative result: representation reconstruction and fluent continuation are both insufficient proxies for functional cache transfer.
>
> My main remaining concern is statistical/procedural rather than experimental. The main text repeatedly describes the \(\epsilon=0.02\) replacement margin as pre-specified and treats the corresponding replacement analysis as confirmatory, whereas Appendix A states that the numeric margin was selected on September 12 after earlier audit runs and is therefore a reporting choice rather than a preregistered constant. These statements must be reconciled. The form of the non-inferiority gate may be confirmatory, but the numeric margin cannot be described as fixed before the relevant runs unless a genuinely untouched evaluation set is used after the margin is chosen.
>
> I would also soften the interpretation of H2: failure of a configuration to clear a non-inferiority gate does not by itself establish inferiority, so “no tested mapper establishes replacement” is preferable to “H2 fails.” Finally, the implementation details of the Heo-style baseline should make explicit how layer selection, held-out selection data, and subsequent refitting are performed.
>
> **Score: 6/10 (Weak Accept), with a likely 7/10 after these consistency fixes.**

---

# 二十三、现在我建议不要再大规模加实验，而是做最后一轮“审稿封口”

当前最优先的不是再加第八个 mapper。

投稿前我会集中处理以下几件事：**第一，把 ε=.02 的时间线彻底统一，删除所有不真实的“pre-specified”表述；第二，把 H2 从“fails”改成“no tested mapper establishes replacement”；第三，把 Conclusion 的“one configuration”修成当前真实的两个 near-chance students / 三个 configurations；第四，把 Figure 1 的 capability upper bound、H1 foreign cache 等旧措辞全部统一；第五，在 Appendix 补 Heo-style baseline 的 selection/refit 细节。**

如果这些完成，我认为论文已经没有明显的结构性审稿漏洞了。

而且从论文价值判断上，我现在会比前几轮更明确：**这已经不是一篇依赖 negative result 才成立的论文。即使未来有人找到一个成功的 cache translator，这篇论文的 H1/H2/H3 audit framework、replacement-vs-gain distinction、PPL/task decoupling、reconstruction/function decoupling 依然成立。**

这正是我认为它现在开始达到 **Accept 档次**的原因。

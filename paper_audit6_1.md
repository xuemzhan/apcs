我把这版 **main(6).pdf** 又按一次“接近正式投稿”的标准完整复审，并重点检查了上一轮遗留问题：\(\epsilon\) 时间线、H2 统计措辞、Heo-style baseline、H3 probe 的逻辑边界、Joint-MLP reconstruction、cross-family controls、PPL 口径和最新相关工作。

这次我的判断比上一轮更明确：

> **科学内容：6.5–7/10，Accept-leaning。**
> **如果按当前稿直接进入严格顶会评审，我会给 6/10 Weak Accept；解决下面两个核心问题后，我会给 7/10 Accept。**
>
> 当前最重要的问题已经不再是“实验够不够多”，而是：
> **Heo-style baseline 是否真的代表 Heo 方法，以及 H3 是否真的能够脱离 mapper 讨论 exploitability。**

---

## 一、上一轮的主要问题，大部分已经真正解决

这版在统计表述上成熟了很多。Abstract 已把 \(\epsilon=0.02\) 明确称作 **reporting replacement margin**，并主动说明数值是在 gate form 冻结以后选择的，不再包装成 preregistered constant。 Section 3.2 现在也正确地区分“gate form 预先固定”与“numerical tolerance 后选”，这是一个很重要的修正。

H2 的主标题也已经从过强的 “H2 fails” 改成：

> **No Strong-Student Mapper Establishes Replacement**

这在统计逻辑上是正确的，因为“没有建立 non-inferiority”不等于“证明 inferiority”。

Heo 对照也变得更诚实。正文现在明确承认自己的实验并非 strict reproduction：Heo 使用 500 条 FineWeb-Edu、每条 1,024 tokens，而且原方法 concatenate selected source layers，而本文用 200 个 audit contexts 并采用 averaging。 

Joint MLP 的 reconstruction diagnostic 依然是这一稿非常强的一部分：更高的 held-out KV \(R^2\) 没有转化成更好的 task performance，这使论文从普通的 negative result 上升成了一个更有普遍意义的结论：

$$
\text{state reconstruction fidelity}\not\Rightarrow\text{functional equivalence}.
$$



所以整体上，现在论文主体是成立的。

---

# 二、当前最重要的问题：你测试的其实还不能叫“Heo baseline”

这是我目前认为**最可能被强 Reviewer 抓住、甚至影响 6→7 分的技术问题**。

原始 Heo et al. 方法不是把 top-k source layers 做平均，而是：

$$
[X_{\ell_1};X_{\ell_2};\cdots;X_{\ell_k}]
$$

即 **concatenate 多个 source-layer KV 作为 ridge 输入**；同时其 calibration 使用 500 条 FineWeb-Edu sequences，每条 1,024 tokens。([arXiv][1])

而本稿现在非常明确地写：

> reference：concatenation
> this audit：average of selected layers



这两个方法的函数空间并不一样。

对于 concatenation：

$$
\hat y
=
W_1x_{\ell_1}
+
W_2x_{\ell_2}
+\cdots+
W_kx_{\ell_k}
$$

每个 source layer 有独立权重。

而 averaging 后：

$$
\hat y
=
W\left(\frac{1}{k}\sum_i x_{\ell_i}\right)
$$

不同 layer 的信息在进入 mapper **之前已经混在一起了**。

这会直接导致一个问题。

你现在观察：

$$
R_V^2:
0.317\rightarrow0.248\rightarrow0.182
$$

随着 \(k=1\rightarrow3\rightarrow5\) 下降，然后解释：

> “combining the selected source layers dilutes the value signal”



但这很可能恰恰是 **averaging implementation** 造成的。

它不能用于解释为什么 Heo 的 multi-source concatenation 方法会失败。

### 因此建议二选一

最优方案是把这项实验真正补完整：

$$
\boxed{\text{top-k concatenation + de-RoPE + ridge}}
$$

最好再尽量匹配：

$$
500\times1024\text{-token FineWeb-Edu}
$$

的 calibration protocol。

如果资源不允许，那么不要再称：

> “implementation of the closest published strict-zero-reprefill design”

而统一改成：

> **“Heo-inspired top-k/de-RoPE ablation under our audit regime”**

同时删除或者弱化：

> “the closest published design fails the same way”

因为现在真正被验证的是**两个 design ingredients**，不是该 published method 本身。

这是我当前最强烈的一条实验建议。

---

# 三、第二个核心问题：H3 仍然没有真正做到“mapper-independent”

论文现在对这一点已经比第一稿谨慎很多，Section 4.4 明确说：

> probes are diagnostic interventions rather than universal upper bounds。



这很好。

但 Introduction 仍然写：

> H2’s null result routes the audit to H3 ... “so a null result cannot be blamed on the mapper.”



严格来说，这句话仍然不成立。

因为 fraction probe 实际做的是：

$$
C(\alpha)
=
\alpha\hat C_S +(1-\alpha)C_S
$$

其中：

$$
\hat C_S=g(C_T)
$$

仍然来自某个 translator。

如果 \(g\) 没有把 Teacher 中真正可利用的结构保留下来，那么：

$$
\Delta(\alpha)\le0
$$

完全可能只是说明：

> **这个 \(g\) 生成的 teacher-derived representation 无法帮助 student**

而不是：

> Teacher cache 本身没有 exploitable content。

你们使用 RAT、affine 两个 source，以及 direct native-content control，确实使这个 objection 弱了很多，但仍然不能彻底消除。

尤其 direct native cache 本身又处于不同 coordinate system，并仍需要固定 depth-ratio layer correspondence，因此也不是“perfect content oracle”。

### 我建议把 H3 再收窄半步

不要定义成：

> “Does the teacher cache contain any student-readable advantage at all?”

建议改成：

> **“Does tested teacher-derived cache content provide incremental benefit when introduced into an otherwise native student cache?”**

或者：

> **“Is teacher-derived cache content exploitable under controlled cache interventions?”**

对应地，把：

> “cannot be blamed on the mapper”

改成：

> **“is not specific to the end-to-end replacement score of a single mapper.”**

这样 H3 就完全站得住了。

甚至我会考虑最终把 “Oracle Probes” 改成：

> **Teacher-Content Probes**

“oracle”这个词本身会让 Reviewer期待一个真正意义上的 upper-bound instrument，而你现在其实做的是非常好的 diagnostic intervention。

---

# 四、最新相关工作仍然缺失，这次我认为不能再不补

我额外核查了截至 **2026-09-13** 的最新文献。

当前 PDF references 仍截止到 Heo / MoT 等 21 篇文献，没有出现两篇与你高度相关的新工作。

第一篇是 9 月 1 日的 **CacheBridge: Efficient Cross-Model KV Cache Transfer**。它针对 Full-Head Mapping 的 architecture mismatch，引入 matched-head support 和 attention-aligned calibration，在 Qwen3 上报告 **99.83% mean target retention**，且使用更少 calibration data。([arXiv][2])

这和你当前的：

* key/value asymmetry；
* architecture mismatch；
* attention routing；
* calibration sensitivity；

高度相关。

第二篇更重要，是 8 月 31 日的 **A Universal Context-Reuse Layer for Cross-Model KV Sharing**。

它报告：

$$
\text{Qwen2.5-7B}\rightarrow\text{Qwen2.5-1.5B}
$$

时，LongBench2 accuracy：

$$
27.59\%\rightarrow34.48\%
$$

也就是 translated KV **高于 target native baseline 6.89 个百分点**。([arXiv][3])

这实际上就是你论文 H3 所关心的：

$$
\text{teacher-derived gain}>0
$$

的一个外部 positive example。

这并不会削弱你的论文。

反而可以显著增强定位：

> **现有文献已经同时出现 severe degradation、high retention 和 above-receiver-baseline gain，因此迫切需要一个能区分 mechanics、replacement、capability gain 的统一审计框架。**

我认为这是比现在的：

> recent methods claim quality-preserving translation → 我们 audit 后发现失败

更强的 Introduction story。

---

# 五、H3 的 family-wise 分析已经是加分项，但统计措辞还可再稳一点

当前你们对 6 个 confirmatory probe configuration 做：

$$
T^{*(b)}=\max_j\bar\Delta_j^{*(b)}
$$

并共享 bootstrap sample indices，保留不同 configuration 之间的相关性。

这比以前逐个 CI 看显著性严格很多。

得到：

$$
U_{95}=+0.009
$$

然后正文写：

> “Any positive average gain larger than about one gold-probability point is therefore excluded at 95% confidence...”



我基本认可这个方向。

但如果 Reviewer 是严格的统计背景，他可能会追问：

> 这是 percentile bootstrap of max，还是 centered max-statistic / max-t？

如果只是：

$$
Q_{.95}\left(\max_j\hat\Delta_j^*\right)
$$

我建议稍微弱化成：

> **“the bootstrap 95% upper estimate for the maximum gain is +0.009.”**

如果想保留“simultaneous 95% exclusion”这种更强措辞，可以做 centered max bootstrap：

$$
Q_{.95}
\left[
\max_j(\hat\Delta_j^*-\hat\Delta_j)
\right]
$$

再构造 upper bound。

这是统计精修，不是目前的拒稿点。

---

# 六、Section 3.3 还有一句最好降级

现在仍然写：

> searching more families, pairs, tasks, and budgets and still finding none “makes the null more informative, not less.”



从经验上我理解作者的意思，但严格统计上：

> 搜索范围更宽

本身不自动增强 null evidence。

还需要结合：

* power；
* effect-size bound；
* sample size；
* simultaneous inference。

建议写：

> **“The broader search reduces concern that the negative result is specific to one hand-picked configuration; the confirmatory probe family is quantified separately with a joint bootstrap analysis.”**

更稳。

---

# 七、还有几处现在已经属于“小修”，但建议投稿前一次性清掉

Figure 1 caption 仍然把 Teacher full prefill 称为：

> **capability upper bound**



正文已经改成 reference，所以这里也应改成：

> **teacher full-prefill reference**

没有必要继续留一个数学上不成立的 upper-bound 表述。

Section 4.4 还有一句：

> “every configuration keeps the student's own cache dominant or localized”

但当：

$$
\alpha=0.75
$$

时 translated content 已占 75%，Student cache 显然不是 dominant。建议改成：

> **“retains a controlled fraction of the student's cache or localizes the intervention to selected layers.”**

另一个值得改的是 Section 5.6：

> “the mapped keys steer attention to plausible but task-irrelevant positions.”



目前没有看到 attention-map analysis 直接证明“task-irrelevant positions”。

所以这里应该和后一句保持一致，改成：

> **“The observed asymmetry is consistent with key-side routing errors.”**

除非再测：

$$
\text{attention-map divergence / target-token attention}
$$

否则不要把 mechanism 写成事实。

还有一句：

> “RAT’s role ... is to rule out the objection that failure is an artifact of insufficiently architecture-aware translation.”



现在已经有 CacheBridge 这种更加 architecture-aware 的 concurrent work，因此建议改成：

> **“RAT tests whether one architecture-informed translator changes the verdict.”**

“rule out”过强。

---

# 八、cross-family 的“non-inferior”最好也与 reporting margin 口径统一

正文现在仍然写：

> “Llama-3.2-1B ... is non-inferior but gains nothing”



但由于：

$$
\epsilon=.02
$$

现在已经明确只是 later-selected reporting margin，所以最好统一写：

> **“clears the reported \(\epsilon=.02\) replacement margin but gains nothing.”**

这样不会让人误解这是预先定义、正式 confirmatory 的 non-inferiority conclusion。

Table 3 也可以把列标题：

> Gate

换成：

> **ε=.02 reporting gate**

更透明。

---

# 九、Section 6 现在的方向是正确的，而且比以前明显更强

我很赞成现在的标题：

> **Why Reconstruction Is Not Enough**



这比以前：

> Why Translation Fails

科学上高了一个层级。

因为你现在真正建立了两组很漂亮的“不等价关系”：

$$
\boxed{
\text{KV reconstruction quality}
\not\Rightarrow
\text{task quality}
}
$$

和：

$$
\boxed{
\text{perplexity / fluency}
\not\Rightarrow
\text{task quality}
}
$$

我认为最终论文应该把这两个 insight 当作中心结果，而不是继续把：

> “Teacher advantage is weight-mediated”

放在最中心。

后者仍然只是：

> evidence is consistent with...

而前两个是实验直接支持的。

---

# 十、当前版本还有几个纯写作问题

第 2 页有：

> “**no** tested strong-student mapper establishes replacement”

句首没有大写。

后面一句：

> “mechanics passes, no strong-student mapper establishes replacement and is inconclusive for the weak one”

主语关系不清楚。

建议拆成：

> *Mechanics passes. No tested mapper establishes replacement for the strong student, while the weak-student result remains inconclusive.*

这种问题虽然不影响科学结论，但目前论文已经接近投稿状态，值得做一次全文 consistency / copy-edit。

---

# 十一、我的最新评分

| 维度                           |   上一稿 |          本稿 |
| ---------------------------- | ----: | ----------: |
| Novelty                      |   8.0 |     **8.0** |
| Technical quality            |   8.5 |     **8.5** |
| Experimental rigor           |   8.5 |     **8.5** |
| Statistical rigor            |   7.5 | **7.5–8.0** |
| Related work                 |   7.5 |     **6.5** |
| Claim–evidence alignment     |   8.5 |     **8.5** |
| Reproducibility              |   9.0 |     **9.0** |
| Clarity                      |   8.5 |     **8.5** |
| Significance                 |   8.5 |     **8.5** |
| **Overall scientific score** | **7** |     **6–7** |

为什么 Related Work 反而降了一点？

不是旧内容变差，而是 **8 月 31 日和 9 月 1 日的新论文已经直接改变了这个领域的证据图景**。如果当前时间投稿，却不引用这两篇高度相关工作，熟悉领域的 Reviewer 很可能会指出 omission。

---

# 十二、如果我是当前轮正式 Reviewer，我会这样写

> **Weak Accept / Accept.**
>
> The manuscript has matured substantially. The authors now distinguish the pre-frozen form of the non-inferiority gate from the subsequently selected numerical reporting margin, correctly scope replacement claims, provide margin sensitivity, report token-aligned cross-family controls, and strengthen the central empirical finding that neither reconstruction fidelity nor near-native perplexity is sufficient for functional cache transfer.
>
> My main remaining technical concern is the comparison to the closest published strict-zero-prefill method. The manuscript now transparently acknowledges that its “Heo-style” implementation differs materially from the reference method: the published mapper concatenates top-k source-layer states and is calibrated on 500 1,024-token FineWeb-Edu sequences, whereas the audit averages source layers and uses 200 task-domain contexts. These differences change the hypothesis class substantially, so this experiment should not be treated as a reproduction of the published baseline. An exact concatenation-based implementation would materially strengthen the paper; otherwise the claims should be framed as a Heo-inspired ablation.
>
> A second conceptual concern is that the H3 probes remain dependent on the teacher-to-student content transformation. Negative mixtures constrain the tested teacher-derived representations, but they do not make exploitability genuinely mapper-independent. I recommend narrowing H3 accordingly and removing the statement that a null probe result “cannot be blamed on the mapper.”
>
> Finally, the related-work section should incorporate very recent cross-model KV-transfer results reporting both near-native target retention and gains over native receiver baselines, which strengthen rather than weaken the motivation for the proposed three-stage audit.
>
> **Score: 6/10 Weak Accept; 7/10 if the baseline and H3 framing are tightened.**

---

# 十三、如果目标仍然是 ICLR 2027，还有两个“与科研质量无关但会直接致命”的问题

当前 PDF 主文仍然一直到第 13 页才进入 References。

而 ICLR 2027 当前官方要求 **submission main text ≤9 pages**，超页会 desk reject；同时要求双盲匿名。([ICLR][4])

当前首页仍包含作者姓名、邮箱和 CETC affiliation。

所以如果这就是准备上传 OpenReview 的 PDF：

> **当前仍然会有 desk-reject 风险，与论文科学质量无关。**

如果只是内部完整版，则没有问题；最终投稿版需要另做 9-page anonymous version。

---

## 最后的修改优先级

| 优先级     | 修改                                                                                        | 必要性       |
| ------- | ----------------------------------------------------------------------------------------- | --------- |
| **P0**  | Heo：做真正的 top-k **concatenation**，或彻底改成 “Heo-inspired ablation”                            | **强烈建议**  |
| **P0**  | H3 删除 “null cannot be blamed on mapper”，收窄 exploitability 定义                              | **强烈建议**  |
| **P0**  | 补 CacheBridge + Universal Context-Reuse Layer                                             | **投稿前应补** |
| **P1**  | Figure 1 的 capability upper bound、cross-family “non-inferior”、RAT “rule out” 等 wording 收口 | 应改        |
| **P1**  | H3 family-wise bound 改 centered max-bootstrap，或弱化统计措辞                                     | 建议        |
| **P2**  | key-routing causal wording、Section 3.3 null wording、全文 copy-edit                          | 小修        |
| **投稿项** | 若投 ICLR：9 页主文、匿名、检查匿名代码仓库                                                                 | **必须**    |

我的总体判断没有倒退：**这篇论文已经达到“值得被顶会认真评审”的水平。** 现在唯一真正可能被技术 Reviewer 持续追问的实验问题，就是你目前所谓的 Heo baseline 与原方法在 **concatenation 和 calibration regime** 上仍存在实质差异。

如果这个问题真正补掉，同时把 H3 从“teacher cache intrinsic exploitability”改成更准确的“tested teacher-derived content benefit”，我会比较稳定地给 **7/10 Accept**。

[1]: https://arxiv.org/abs/2608.03893?utm_source=chatgpt.com "Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse"
[2]: https://arxiv.org/abs/2609.00891?utm_source=chatgpt.com "CacheBridge: Efficient Cross-Model KV Cache Transfer"
[3]: https://arxiv.org/abs/2608.30963?utm_source=chatgpt.com "A Universal Context-Reuse Layer for Cross-Model KV Sharing"
[4]: https://iclr.cc/Conferences/2027/AuthorGuidelines?utm_source=chatgpt.com "ICLR 2027 Author Guidelines"

我把这版 **19 页最新稿（main(5).pdf）**重新按顶会正式审稿标准完整复审了一遍，并额外核查了截至 2026 年 9 月中旬最新出现的跨模型 KV-cache 工作。

这一版继续明显进步。上一轮最关键的几项修改——H1 定义、H2 标题、\(\epsilon\) 时间线披露、margin sensitivity、Heo-style 实现说明、Joint MLP reconstruction、family-wise H3 分析、Gold probability 定义——基本都进入正文或附录了。

我的最新判断是：

> **科学内容：7/10，Accept 倾向。**
> **当前文稿状态：6/10，Weak Accept。**
>
> 现在已经没有需要“推翻实验框架”的问题。剩余风险主要集中在三个地方：
> **① \(\epsilon=.02\) 仍有几处前后矛盾；② Heo baseline 还不能称为严格复现；③ 最新相关工作已经出现，必须补。**
>
> 如果这三项处理好，我会稳定给 **7/10 Accept**。

---

# 一、先说结论：论文主体现在已经站住了

这一版最重要的变化，是论文真正形成了一个完整的 audit paper，而不再依赖“KV translation 是不是不可能”这个强结论才能成立。

现在最扎实的三个结果是：

$$
\boxed{\text{Injection mechanics work}}
$$

identity control 得到 logit cosine 0.99998，而且 cold-start persistence 路径也做到逐样本完全复现。

第二个结果是：

$$
\boxed{\text{Better KV reconstruction}\not\Rightarrow\text{better task performance}}
$$

per-head affine 的 held-out \(R^2\) 是：

$$
K=0.81,\quad V=0.32
$$

position-wise Joint MLP 提升到：

$$
K=0.854,\quad V=0.481
$$

但 downstream CHG 反而从约 \(-0.14\) 下降到约 \(-0.27\)。

第三个仍然是非常漂亮的：

$$
\boxed{\text{Fluency}\not\Rightarrow\text{capability}}
$$

PPL 已经恢复到 23.7，接近 identity 21.2，但 accuracy 仍然只有 0.267，而 student 是 0.500。

这三个结果组合起来，已经足以构成论文的核心贡献。

---

# 二、上一轮最严重的 \(\epsilon=.02\) 问题，解决了 80%，但还没有完全解决

这是现在**最优先需要修改的内部一致性问题**。

好的一面是，Abstract 最后现在已经非常诚实地写：

> numerical replacement margin \(\epsilon=.02\) 是在 gate form 固定之后才选择的，因此只是 practical reporting tolerance，而不是 preregistered constant。



Appendix A 也完整记录了时间线：

* framework / probe family：8 月 29 日；
* non-inferiority **形式**：8 月 29 日；
* calibration ladder：8 月 30–31 日；
* numerical \(\epsilon=.02\)：9 月 12 日。

因此 Appendix 的统计诚信处理现在很好。

而且还增加了 margin sensitivity：

$$
\epsilon=.01:\quad0\text{ 个 configuration pass}
$$

$$
\epsilon=.02:\quad3\text{ 个 distinct configurations pass}
$$

$$
\epsilon=.05:\quad5\text{ 个 distinct configurations pass}
$$

这正是我上一轮建议增加的内容。

---

# 三、但是正文中仍然残留三个与 Appendix 冲突的句子

### 第一处就在 Abstract

前面仍然写：

> “our **pre-specified** replacement margin (\(\epsilon=.02\))”



但同一个 Abstract 后面又承认它不是 preregistered。

这会让 Reviewer 非常容易发现内部冲突。

直接改成：

> **our reporting replacement margin (\(\epsilon=.02\))**

即可。

---

### 第二处更严重，在 Section 3.2

现在仍然写：

> “We fix \(\epsilon=.02\) gold probability **before evaluation** ...”



这个句子和 Appendix 记录的 commit timeline 明确矛盾。

必须改成类似：

> “We report replacement results at \(\epsilon=.02\), corresponding to roughly 4% of the primary student's gold probability. The non-inferiority gate form was frozen before the audit runs, whereas this numerical tolerance was selected later as a practical reporting margin; Appendix A reports sensitivity to alternative margins.”

这是投稿前**必须修改**的。

---

### 第三处在 Section 7

这里仍然写：

> “neither interval clears the **pre-specified non-inferiority margin**”



同样改成：

> **the reported \(\epsilon=.02\) replacement margin**

即可。

这三处改完，\(\epsilon\) 的问题才算真正关闭。

---

# 四、H2 的统计措辞也还有几个旧版本残留

Section 5.2 标题已经很好地从：

> H2 fails

改成：

> **No Strong-Student Mapper Establishes Replacement**



这是正确的统计逻辑。

但是 Introduction 还保留：

> “H2 fails for the strong student”

以及：

> “replacement fails for the strong student”

和：

> “H2’s failure routes the audit to H3”

 

Section 5.2 后面也还有：

> “H2’s failure is robust to teacher scale.”



这些都建议统一成：

> **No tested strong-student mapper establishes replacement.**

或者：

> **Failure to establish H2 persists at the larger teacher scale.**

原因很简单：

$$
\text{failed to establish non-inferiority}
\neq
\text{established inferiority}
$$

尽管多数配置确实有显著 degradation，但最佳 affine \(c=30\) 的：

$$
[-0.317,+0.029]
$$

既没有建立 replacement，也没有建立 degradation。

所以现在 Section 5.2 的新标题反而比 Introduction 更严谨，建议全文统一。

---

# 五、Heo-style baseline 是很重要的增强，但我现在发现了一个更深的问题

这是本轮复审里我认为**最值得认真处理的技术问题**。

论文现在写：

> implementation of the closest published strict-zero-reprefill design

并实现了：

* top-k cross-layer selection；
* de-RoPE keys；
* ridge；
* \(k=1,3,5\)。

结果全部明显退化。

这个实验非常重要。

但 Appendix 的 implementation table 现在说：

> 原论文 calibration volume “not specified”。



这一点不准确。

Heo et al. 的论文明确写的是 **500 条 FineWeb-Edu sequences，每条 1,024 tokens**。他们也是在这个 calibration regime 下报告 73–98% retention。([arXiv][1])

而你现在用的是：

$$
200\text{ contexts}
$$

并且来自自己的 task/calibration distribution。

这不是一个小差异。

---

# 六、因此现在不能把这项实验表述成“完全覆盖 Heo baseline”

当前 Section 5.2 写：

> “The negative result therefore covers the closest published baseline...”



这句话我建议降级。

因为你真正覆盖的是：

> **Heo 的 top-k selection + de-RoPE 这两个关键 design elements，在本文自己的 calibration regime 下的实现。**

你没有完全复现：

$$
500\times1024\text{-token FineWeb-Edu}
$$

的 calibration protocol。

更准确写法：

> **“The negative result persists after importing the two defining design choices of the closest published strict-zero-reprefill mapper—top-k cross-layer selection and de-RoPE key mapping—into our audit calibration regime.”**

然后 Appendix 表里直接写清：

| Aspect          | Heo et al.                    | This audit                 |
| --------------- | ----------------------------- | -------------------------- |
| calibration     | 500 FineWeb-Edu × 1024 tokens | 200 audit-context examples |
| layer selection | top-k                         | top-k                      |
| key treatment   | de-RoPE                       | de-RoPE                    |

这样 Reviewer 会认为你是公平比较，而不是声称 exact replication。

如果资源还允许，我认为**最有价值的最后一项实验**就是：

$$
500\times1024
$$

或尽量接近 Heo calibration volume/domain，再跑一次 \(k=1/3/5\)。

但这已经不是“论文能不能投稿”的必要条件。

如果不跑，只需把 claim 收窄。

---

# 七、还有一个可能更严重的 Heo implementation 问题：到底是“concatenate”还是“average”？

Heo 原方法的描述是：

> 选择 top-k source layers，然后**把这些 KV concatenated as input** 再拟合 ridge。([arXiv][1])

但你论文 Section 6 对自己的 Heo-style baseline 解释成：

> “**averaging several source layers** dilutes the value signal...”



这里必须确认代码。

如果你的实现实际上是：

$$
[x_{\ell_1};x_{\ell_2};\ldots;x_{\ell_k}]
$$

concatenation，那么正文的 “averaging” 只是写错了，改成：

> **combining / concatenating additional source layers**

即可。

但如果代码真的把多个 layer 做：

$$
\frac1k\sum_i x_{\ell_i}
$$

那这就**不是 Heo 方法**，必须修改实验。

这是我建议作者在投稿前直接检查代码的一项。

---

# 八、好消息是：Heo baseline 的 reconstruction 诊断现在做得很好

这一版补了：

$$
k=1:
R_K^2=.822,\quad R_V^2=.317
$$

$$
k=3:
.787,\quad .248
$$

$$
k=5:
.762,\quad .182
$$

并且 PPL 同步恶化：

$$
78.5\rightarrow2629.6\rightarrow6.9\times10^4
$$



这很好，因为现在可以明确说：

> 在这个 pair / calibration regime 下，Heo-style multi-source design 并不是“reconstruction 很好但 functional behavior 坏掉”，而是 reconstruction 本身就随 k 增大恶化。

这个机制解释比上一版强很多。

只要把 calibration mismatch 和 concatenate/average 搞清楚，这部分会很有说服力。

---

# 九、最新相关工作需要补：现在有两篇非常直接的新论文

我额外核查了 9 月最新论文，这里建议一定更新 Related Work。

## CacheBridge

9 月 1 日公开的 **CacheBridge: Efficient Cross-Model KV Cache Transfer**，直接建立在 Heo-style Full-Head Mapping 上。

它指出 full-head mapper 对 architecture mismatch 很敏感，并使用：

* matched source head；
* causal-attention-weighted calibration；
* bounded mapper construction。

论文声称在 Qwen3 上达到 **99.83% mean target retention**，而且使用约原方法十分之一的 calibration data。([arXiv][2])

这篇与你现在的：

* Heo baseline failure；
* key/value asymmetry；
* reconstruction fidelity；
* attention routing sensitivity

高度相关。

至少必须在 Section 2.2 加入讨论。

---

## A Universal Context-Reuse Layer for Cross-Model KV Sharing

8 月 31 日公开的这篇更值得注意。

它报告 Qwen2.5-7B→1.5B 时：

$$
27.59\%\rightarrow34.48\%
$$

也就是 translated KV 的 LongBench2 accuracy **高于 native 1.5B baseline 6.89 个百分点**。([arXiv][3])

这正好对应你 H3 最关心的：

> teacher-derived gain over student self-prefill。

它并不否定你的论文，因为你的结论已经严格限定为：

> **tested translators / tested regime**

但它非常值得引用，因为它说明：

> positive H3 result 在另一个设计空间可能确实存在。

这反而会强化你的论文定位：

> **audit framework 用来判断什么时候真正发生 capability gain，而不是提前宣判它不存在。**

我建议把这篇写进 Related Work，而不是回避。

---

# 十、这两篇新工作实际上让你现在的论文故事更好

我甚至会建议 Introduction 稍微调整。

不要再把论文主要卖成：

> “recent systems report quality preservation, but我们发现不行。”

更好的 story 是：

> **Recent work now reports outcomes ranging from severe degradation, through near-perfect receiver retention, to gains over the native receiver. These heterogeneous results make it especially important to separate mechanics, replacement, and capability gain under a common audit protocol.**

这样你这篇论文会从：

> negative-result paper

变成：

> **the measurement paper needed because the literature is now giving conflicting outcomes.**

这是更强的 positioning。

---

# 十一、H3 family-wise bound 是明显加分，但统计实现可以再严谨一点

现在 Appendix 已明确写出：

$$
T^{*(b)}=\max_j\bar\Delta_j^{*(b)}
$$

而且所有 configurations 使用同一组 bootstrap indices，保留 sample-level correlation。

这是很好的一步。

不过如果你要使用很强的措辞：

> “Any gain larger than +0.009 is excluded at 95% confidence”



一个统计比较严格的 Reviewer 可能会问：

> 这里是 raw percentile bootstrap of max，还是 centered / studentized max-statistic？

对于 simultaneous one-sided coverage，更标准的是计算：

$$
q_{.95}
=
Q_{.95}
\left[
\max_j
(\hat\Delta_j^*-\hat\Delta_j)
\right]
$$

然后构造 simultaneous upper bound。

目前描述看起来更像直接对：

$$
\max_j\hat\Delta_j^*
$$

取 95 percentile。

这作为 bootstrap interval approximation 可以用，但 n=30、max operator 又是非光滑统计量，最好不要把它包装成特别严格的 FWER theorem。

两种处理都可以：

要么改成更加谨慎的：

> **“the bootstrap 95% upper estimate for the maximum gain is +0.009.”**

要么改用 centered max-bootstrap / max-t。

这是一个**统计精修项，不是核心缺陷**。

---

# 十二、Section 3.3 有一句我建议弱化

现在写：

> 搜索更多 families / pairs / tasks 仍没有 positive result，
> “makes the null more informative, not less.”



从直觉上没错，但作为统计说法稍微过于简化。

因为“没有发现显著正结果”的证据强弱不仅取决于搜索范围，还取决于：

* power；
* sample size；
* effect-size upper bound。

建议改成：

> **“The broader search reduces concern that the negative result is specific to one hand-picked configuration; the confirmatory probe family is quantified separately with a simultaneous bootstrap bound.”**

更加严谨。

---

# 十三、Cross-architecture 部分现在基本关闭了上一轮问题

这版已经给 Llama-3.2-1B 和 Gemma-2-2B 两个 near-margin rows 都补了 token alignment。

结果分别：

$$
-0.003[-0.011,+0.005]
$$

和：

$$
-0.007[-0.018,+0.005]
$$

仍然清掉 \(-.02\) reporting margin，而且 gains 都跨零。

这一部分现在已经足够。

唯一措辞建议：

目前正文还说：

> “Llama-3.2-1B ... **is non-inferior**”



因为 \(\epsilon=.02\) 不是 preregistered margin，更稳妥的说法是：

> **“clears the reported \(\epsilon=.02\) replacement margin.”**

避免把 post-hoc reporting margin 写成正式 confirmatory non-inferiority conclusion。

---

# 十四、Figure 1 还有一个旧词没清理掉

正文现在已经把 teacher 定义成：

> **teacher full-prefill reference**

这是对的。

但 Figure 1 caption 仍写：

> **capability upper bound**



建议改成：

> **teacher full-prefill reference**

因为它并不是数学意义的 upper bound。

Figure 1 关于 \(\epsilon\) 的 § marker 现在倒是处理得很好，已经明确写：

> numerical replacement margin selected later as reporting tolerance。

只需要把 capability upper bound 清掉。

---

# 十五、Section 6 的重命名非常成功

从：

> Why Translation Fails

改成：

> **Why Reconstruction Is Not Enough**



我认为这是这一版非常好的修改。

现在文章的机制解释不再试图证明：

> KV fundamentally cannot work。

而是在证明：

$$
\text{reconstruction fidelity}
\not\Rightarrow
\text{functional equivalence}.
$$

这是强得多也更安全的 claim。

我甚至认为最终 Abstract 可以稍微突出这一点。

---

# 十六、weights 的表述也已经基本过关

现在写的是：

> teacher advantage “is consistent with being mediated by parameter-dependent computation”

而且进一步说：

> raw KV transfer does not **explicitly transport** it。



这个版本已经比最初：

> capability is in weights, not cache

严谨得多。

我认为可以保留。

只需要注意一句：

> “a learned translator can at best track the conditional mean”



最好限定为：

> **“for the per-head squared-error prediction problem, the Bayes-optimal predictor is the conditional mean.”**

否则容易被理解成对所有 nonlinear / joint translators 的 universal claim。

---

# 十七、Cold-start path 现在是一个不错的工程增强

offline cache + mapper 落盘之后，online run：

* 不保留 teacher；
* 逐样本 gold 完全一致；
* max abs difference = 0。



这个实验很好，因为它说明 strict zero-reprefill 不是“只在同一个 Python process 里模拟 cache injection”。

Limitation 也正确承认它只做了一组 paired run。

这里不需要继续补实验。

---

# 十八、论文当前最大的“投稿级”问题可能已经不是科学问题，而是格式

如果目标仍然是 **ICLR 2027**，当前 PDF **不能直接提交**。

ICLR 2027 官方要求：

> main text **9 pages or fewer**；超页将 desk reject。([ICLR][4])

当前论文正文一直到第 13 页 Conclusion，References 从第 14 页开始，因此主文约 **13 页**。

也就是说，需要压缩大约：

$$
13\rightarrow9
$$

页。

这已经不是小修。

建议把以下内容优先移 Appendix：

* cross-architecture 详细结果；
* 1K/4K retrieval；
* RAT component detail；
* channel asymmetry；
* 部分 Section 6 mechanism；
* margin sensitivity；
* text-channel secondary comparison。

主文只留下：

1. H1/H2/H3；
2. Table 2 精简版；
3. Figure 2；
4. Figure 3；
5. reconstruction–task decoupling；
6. PPL–task decoupling。

---

# 十九、如果是 ICLR 2027，还有一个直接 desk-reject 风险

ICLR 2027 是 double blind，而且官方明确说：

> 论文或 supplementary 中泄露作者身份会被 desk reject。([ICLR][4])

当前首页直接有：

* 作者姓名；
* 邮箱；
* CETC。



如果这只是内部审阅版，无所谓。

如果这是准备上传 OpenReview 的版本，必须匿名化。

同时要检查：

* repository；
* commit；
* README；
* code paths；

不能通过公开实名 GitHub 反查作者。

---

# 二十、我的最新评分

| 维度                       |     上一稿 |            当前稿 | 评价                                      |
| ------------------------ | ------: | -------------: | --------------------------------------- |
| Novelty                  |     8.0 |        **8.0** | audit framing 已成熟                       |
| Technical quality        |     8.0 |        **8.5** | reconstruction/function 解耦增强            |
| Experimental rigor       |     8.5 |        **8.5** | 当前实验矩阵已经足够丰富                            |
| Statistical rigor        |     6.5 |        **7.5** | margin sensitivity 与 max-bootstrap 明显进步 |
| Related work             |     8.5 |        **7.5** | Heo 处理更细，但缺 8/31–9/1 两篇最新直接工作           |
| Claim–evidence alignment |     8.0 |        **8.5** | 主结论已经很克制                                |
| Reproducibility          |     9.0 |        **9.0** | 时间线、commit、seed、重跑记录非常好                 |
| Clarity                  |     8.5 |        **8.5** | Section 6 重构尤其好                         |
| Significance             | 8.0–8.5 |        **8.5** | 现在更像领域 evaluation methodology           |
| Overall scientific score |       6 | **7 / Accept** | 前提是修掉下面几个明确问题                           |

---

# 二十一、如果我是 Reviewer，现在正式意见会是

> **Accept.**
>
> This revision resolves most of my previous concerns. The paper now cleanly separates the preregistered form of the non-inferiority gate from the subsequently selected numerical reporting margin, reports margin sensitivity, adds token-aligned controls for all cross-tokenizer students, provides a cold-start persistence test, and substantially strengthens the mechanistic analysis. Particularly compelling is the observation that a larger joint mapper improves held-out KV reconstruction but worsens downstream performance, providing direct evidence that state reconstruction fidelity is not sufficient for functional cache transfer. Together with the perplexity–accuracy dissociation, this makes the work substantially more informative than a generic negative result.
>
> I have two remaining substantive concerns. First, several main-text sentences still incorrectly describe \(\epsilon=0.02\) as pre-specified or fixed before evaluation, despite Appendix A correctly documenting that the numerical margin was chosen later. These should be made fully consistent.
>
> Second, the “Heo-style” experiment imports important components of the published method but does not match its reported calibration regime. The reference work uses 500 FineWeb-Edu sequences of 1,024 tokens, whereas the audit uses 200 contexts. The manuscript should therefore characterize this experiment as an implementation of the key top-k/de-RoPE design choices under the audit regime rather than as a full reproduction, unless a calibration-matched experiment is added.
>
> I also encourage the authors to update the related work with very recent cross-model KV transfer systems reporting both near-native retention and above-receiver-baseline gains.
>
> **Score: 7/10 – Accept.**

---

# 二十二、现在我只建议做最后一轮“投稿封口”，不要再无限加实验

按优先级，我建议最后只处理这 **8 件事**：

1. 把 Abstract 的 **pre-specified \(\epsilon=.02\)** 改成 **reporting margin**。
2. 把 Section 3.2 的 **“fix ε=.02 before evaluation”** 删除。
3. 把 Section 7 的 **pre-specified margin** 改成 reporting margin；cross-family 的 “non-inferior” 改成 “clears the reported margin”。
4. 把 Introduction 中残余的 **H2 fails / replacement fails / H2's failure** 全部改成 **does not establish replacement**。
5. 修 Figure 1 的 **capability upper bound → teacher full-prefill reference**。
6. 核查 Heo-style 代码到底是 **concatenate 还是 average**，并把 calibration 差异如实写成 **500×1024 FineWeb-Edu vs 200 audit contexts**；如果不完全复现，不要写“covers the published baseline”。
7. 补 **CacheBridge** 和 **Universal Context-Reuse Layer** 两篇最新相关工作；尤其后者已经报告 translated KV 高于 receiver native baseline 的 positive gain。([arXiv][3])
8. 如果目标是 ICLR 2027，立即处理 **9 页主文限制 + 双盲匿名**，否则科学上能过、形式上会直接 desk reject。([ICLR][4])

完成这些以后，我不建议再继续堆 mapper 或继续扩展实验矩阵。

现在这篇论文的价值已经不依赖“所有 KV translation 都失败”这个命题。恰恰相反，随着最新工作开始同时报告 **失败、近乎完整 retention、甚至 receiver-baseline 以上的 gain**，你提出的 **mechanics → replacement → capability gain** 三层 audit framework 反而更有必要。

**所以我的最终判断已经从前几轮的 Borderline / Weak Accept，提升为：科学内容达到 Accept 水平；当前稿件再做一轮一致性和投稿格式收口即可。**

[1]: https://arxiv.org/abs/2608.03893?utm_source=chatgpt.com "Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse"
[2]: https://arxiv.org/abs/2609.00891?utm_source=chatgpt.com "CacheBridge: Efficient Cross-Model KV Cache Transfer"
[3]: https://arxiv.org/abs/2608.30963?utm_source=chatgpt.com "A Universal Context-Reuse Layer for Cross-Model KV Sharing"
[4]: https://iclr.cc/Conferences/2027/AuthorGuidelines?utm_source=chatgpt.com "ICLR 2027 Author Guidelines"

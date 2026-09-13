我把最新的 **main(8).pdf** 按“正式投稿前最后一轮审稿”的标准重新完整审了一遍，并重点对照前几轮意见检查了：Heo baseline、H3 定义、最新相关工作、统计 gate、Identity Control、长上下文实验、Joint MLP、论文压缩后的信息损失以及图表一致性。

这版是目前为止**最成熟的一版**。上一轮两个最重要的问题——“Heo 方法只做 average 而没有 concatenate”以及“H3 过度声称 mapper-independent exploitability”——都已经实质解决。

我的当前结论是：

> **Overall：7/10，Accept。Reviewer confidence：4/5。**
>
> 如果现在进入 ICLR/NeurIPS/ICML 级别同行评审，我会投 **Accept / Weak Accept toward Accept**。
> 当前已经没有我认为足以导致拒稿的核心方法论漏洞。
>
> 但投稿前仍有 **2 个值得认真修的统计/方法问题 + 约 6 个容易修的 consistency 问题**。这些问题不需要再扩实验矩阵。

---

## 一、这版已经解决了哪些关键问题

最重要的变化，是你现在没有再把“我们的方法没有成功”包装成“KV 本质不可能”。

Abstract 已经非常克制：

> “Under zero re-prefill, we find no evidence that the teacher’s answer-relevant advantage is recoverable from its cache by the tested translators.”

同时 H3 现在明确改成：

> “tested teacher-derived cache content provides an incremental benefit when introduced into an otherwise native student cache.”

而不是过去的：

> teacher cache 是否 intrinsically contains exploitable advantage。

这正是我上一轮最希望看到的修改。 

第二个重大改进是 Heo baseline。

现在你不仅承认差异，而且真正加入了：

$$
\text{top-k source layers}
+
\text{concatenation}
+
\text{de-RoPE}
+
1024\text{-token web-text calibration}
$$

结果：

$$
k=1:-0.279
$$

$$
k=3:-0.223
$$

$$
k=5:-0.254
$$

三个 CI 全部低于零。 

更重要的是 Appendix 非常诚实地写明：

> reference 使用 500 条 FineWeb-Edu；本文使用 100 条 generated web-text-style sequences。

没有再把它包装成完全复现。

这一修改基本关闭了我上一轮最大的技术 objection。

第三，最新相关工作已经补进来了：

* CacheBridge；
* Universal Context-Reuse Layer。

而且 Introduction 的定位明显比以前更强：

> literature 现在同时存在 near-native retention、receiver-baseline 以上 gain，以及 severe degradation，因此需要统一 audit framework。

这是现在最合理的论文定位。 

我认为这个 narrative 已经成立。

---

# 二、我现在怎么看这篇论文的核心贡献

这篇论文现在真正有价值的已经不是：

> “cache translation 不行”。

而是三个相对稳定、即使以后出现更强 translator 仍然成立的贡献。

第一：

$$
\boxed{\text{Mechanics} \neq \text{Mapping} \neq \text{Exploitability}}
$$

这个 H1/H2/H3 分解现在很干净。

第二：

$$
\boxed{
\text{KV reconstruction fidelity}
\not\Rightarrow
\text{functional equivalence}
}
$$

Joint MLP 把 held-out reconstruction 提高到：

$$
R_K^2=0.854,\qquad R_V^2=0.481
$$

但 downstream CHG 仍然：

$$
-0.30\sim-0.26
$$

而简单 affine 的 CHG 最好反而是：

$$
-0.138.
$$

 

第三：

$$
\boxed{
\text{Fluency}
\not\Rightarrow
\text{Capability}
}
$$

PPL：

$$
23.7
$$

已接近 identity：

$$
21.2
$$

但 accuracy：

$$
26.7\%
$$

对比 student：

$$
50.0\%.
$$

Figure 3 已经把这个结果表现得很直观。

这两个 “\(\not\Rightarrow\)” 是现在论文中我认为最值得留下来的 punchline。

---

# 三、当前最值得修的第一个问题：H1 的“exactly”仍然略过头

Abstract 现在写：

> “Mechanics passes exactly”

但对应结果其实是：

$$
\text{logit cosine}\ge0.99997
$$

主 pair：

$$
0.99998
$$

gold CHG：

$$
+0.0001[-0.0006,+0.0009].
$$



Figure 1 的 formal gate 仍然写：

$$
\text{per-sample logit cosine}=1
$$

而真实结果不是数学意义上的 1。

这其实很容易被一个较真的 Reviewer 抓住：

> 0.99998 为什么叫 “within float tolerance”？

float32 machine epsilon 本身远小于 \(2\times10^{-5}\)，所以“float tolerance”这个词并不严谨。

### 建议

把：

> **Mechanics passes exactly**

统一改成：

> **Mechanics passes to numerical tolerance**

然后把 H1 gate 定量定义出来，例如：

$$
\cos(z_{\rm inj},z_{\rm native})\ge0.9999
$$

并最好再报告：

$$
\max_i\|z^{(i)}_{\rm inj}-z^{(i)}_{\rm native}\|_\infty
$$

这样 H1 会更加硬。

尤其论文现在把 identity control 定位成未来领域 checklist，这个 gate 最好是一个可复现的数值标准，而不是：

> “=1 within float tolerance”。

这是我当前最建议修的 methodology detail。

---

# 四、第二个比较重要的问题：replacement gate 的三个标签并不天然互斥

当前 Table 1 的定义是：

> pass：CI lower bound \(>-\epsilon\)
> deg.：CI 全部低于 0
> n.e.：lower \(\le-\epsilon\)，upper \(\ge0\)



这里存在一个统计上的一般性问题。

考虑：

$$
CI=[-0.015,-0.005]
$$

且：

$$
\epsilon=0.02.
$$

那么：

$$
-0.015>-0.02
$$

所以按照 non-inferiority gate：

> replacement **pass**

但同时：

$$
CI_{\rm upper}<0
$$

又意味着：

> 相比 baseline 有显著负差异。

也就是说：

$$
\text{statistically worse}
$$

和：

$$
\text{practically non-inferior}
$$

完全可以同时成立。

这不是矛盾，而是标准 non-inferiority 统计中很正常的结果。

所以：

> pass / deg / n.e.

严格来说不应该被设计成一个 mutually-exclusive 单列标签。

当前数据刚好似乎没有出现这一交叉情况，因此不影响你现在任何实质结论。

但作为论文提出的 **general audit protocol**，最好修掉。

### 最好的设计

把 Table 1 的一个 “Gate” 列拆成两个判断：

| Replacement | Direction                 |
| ----------- | ------------------------- |
| pass/fail   | gain / neutral / degraded |

其中：

$$
\text{Replacement pass}
\iff CI_L>-\epsilon
$$

$$
\text{Gain}
\iff CI_L>0
$$

$$
\text{Significant degradation}
\iff CI_U<0.
$$

这样统计逻辑最干净。

如果版面实在不允许增加一列，至少在 Section 3.2 加一句：

> *Non-inferiority and significant difference from zero are conceptually separate; a confidence interval can establish both practical non-inferiority and a small statistically significant degradation.*

这是我认为当前版本唯一一个真正值得修改的“框架级”细节。

---

# 五、Heo-style baseline 现在够用了，但还能再精确一点

这一版已经明显比上一版可靠。

Method 很清楚地说：

> average variant：200 audit contexts；
> concatenate variant：100 条 1,024-token generated web-text-style sequences；
> reference：500 sequences。



而 Appendix Table 2 也诚实列出差别。

所以我现在**不会要求必须补 500 条 FineWeb-Edu 才能接收**。

但有两个措辞还可以再谨慎一点。

Section 5.2：

> “The closest published design fails in both forms we can match—its source-layer combination and its sequence length...”

这个是可以接受的。

但 Conclusion：

> “it is not repaired by matching the closest published design’s composition and calibration length.”



这里的 “matching” 容易让人觉得 protocol 已完全对齐。

更准确：

> **“matching its source-layer composition and calibration sequence length under our reduced synthetic calibration corpus.”**

因为你没有匹配：

* calibration volume；
* FineWeb-Edu distribution。

如果还有资源，最有价值的唯一 baseline 扩展仍然是：

$$
100\rightarrow500
$$

条真实 FineWeb-Edu。

但我现在把它归为 **nice-to-have，而不是 acceptance blocker**。

---

# 六、论文压缩之后出现了一个新的内部 consistency 问题：到底测试了多少 mapper families？

Method 现在列出：

* per-head ridge；
* per-head affine；
* per-layer affine；
* **low-rank shared-basis maps**；
* task-aware；
* RAT；
* per-head MLP；
* joint MLP。



按照自然理解，这是 **8 类**。

但 Table 1 没有明显看到一个独立的：

> low-rank shared-basis mapper

行。

`RAT no-anchor rank-32` 可能是你想指的 low-rank configuration，但它看起来更像 RAT ablation，而不是独立 mapper family。

这会让 Reviewer 问：

> 你 Method 说 test 了 low-rank shared-basis maps，它的数据在哪里？

建议二选一：

* 如果 RAT no-anchor rank-32 就是这个 family，正文直接说明；
* 如果不是，就把 “low-rank shared-basis maps” 从 mapper-family enumeration 中删掉。

这是一个小问题，但当前论文已经非常紧凑，越容易让人察觉这种计数不一致。

---

# 七、Abstract 有一个明确的数量错误

Abstract 写：

> “on Qwen3-4B→1.7B and **four further scale pairs**”



但 Method 实际列出：

* Qwen3-4B→1.7B primary；
* 4B→0.6B；
* 8B→1.7B；
* 8B→0.6B。



所以应该是：

> **four Qwen3 scale pairs in total**

或者：

> **the primary pair and three additional Qwen3 scale pairs**

不是 four further。

这个必须改。

---

# 八、H3 现在基本站得住了，但 Section 7 还有一句略微超出证据

H3 的正式定义现在很好：

> tested teacher-derived content / controlled interventions / not intrinsic information content。



这已经解决了上一版的核心逻辑问题。

但 Section 7 仍然写：

> “Report a training-free probe of teacher-derived content before investing in translator capacity, **because no translator we test recovers a gain the probe did not already reveal**.”



这句话容易被读成：

> probes 可以 upper-bound translator。

但实际上 probes 的 content 仍来自 RAT/affine 等 translator。

建议改成：

> **“because in our audit the probes reveal no latent benefit that is hidden by the end-to-end replacement score.”**

或者更简单：

> **“because they provide an independent diagnostic of whether introducing tested teacher-derived content helps an otherwise native student cache.”**

这样完全不承担 oracle/bound 意义。

---

# 九、Section 5.3 “teacher-content probes return nothing”可以更加学术一点

现在标题是：

> **H3: teacher-content probes return nothing**



我理解这种标题很有力量。

但实际上 probes 返回了很多有用信息：

* fraction 越大 degradation 越明显；
* top third neutral；
* bottom/middle harmful；
* octant 存在 +0.020 非显著点估计；
* RAT 与 affine 方向一致。

所以严格说不是 “return nothing”。

建议改成：

> **H3: No Teacher-Content Probe Establishes Gain**

这样与整篇统计语言也更加一致。

---

# 十、Family-wise 统计这一版处理得很好

上一轮我专门指出 max-bootstrap 不能随便写成 simultaneous 95% exclusion。

这一版已经明确：

> observed maximum = −0.001；
> 95% upper estimate = +0.009；
> 这是 percentile bound on a maximum；
> **不是 centered max-t construction；**
> 所以只解释成：
>
> “bootstrap 95% upper estimate of the best achievable gain in this family is about one gold-probability point”。



这是很成熟的修改。

我认为这一点已经不用再改。

---

# 十一、Context-length 部分现在比前稿更强，但仍应保持现在这种克制

你们现在新增了真正的：

$$
4K,\quad8K
$$

并且结果都是明显负：

$$
4K:-0.596[-0.929,-0.170]
$$

$$
8K:-0.577[-0.932,-0.118].
$$



但是你们也非常正确地写明：

* \(n=10\)；
* float16 cache；
* fixed-tail protocol disabled；
* evaluation regime 与 1K 不同；

所以只把它解释为：

> further degradation evidence，而不是 calibrated length sweep。



这部分现在处理得很好。

不要再为了“8K”去扩到 16K/32K。

当前已经够了。

---

# 十二、Channel asymmetry 的表述也已经修到合理程度

以前的：

> mapped keys route attention to task-irrelevant positions

是一个没有 attention analysis 支持的因果判断。

现在改成：

> “we do not claim to have identified the mechanism”
> “the reading we consider most consistent ... key-side routing errors.”



这已经很好。

可以保留。

---

# 十三、一个新出现的 reproducibility 风险：same-seed MLP 不复现

Limitations 很诚实地写：

> two runs with the same configuration and same recorded seed differ by 0.046 in CHG。

因此 per-head MLP 只能报告 range，而不能作为 reproducible point estimate。

从诚信角度，这是加分。

但 Reviewer 会自然问：

> 为什么同 seed 不 deterministic？

最好在 Appendix B 给一句原因候选，例如：

* CUDA nondeterministic kernels；
* mixed precision；
* dataloader order；
* library-level nondeterminism。

如果你不知道原因，就明确写：

> “the residual source of nondeterminism has not been isolated.”

我不建议为了这个问题重新跑大量实验，因为你已经正确把 gradient-trained family 降格成 range evidence。

但是对一篇主打“audit / reproducibility”的论文来说，最好说明：

> **deterministic algorithms 是否开启。**

这是一个小但很值得补的 reproducibility detail。

---

# 十四、Bootstrap 主结果的实现细节压缩得稍微过头了

最新 13 页版本为了减篇幅，把之前 Appendix 里比较完整的统计协议基本删掉了。

正文现在只说：

> “95% bootstrap intervals”



H3 family-wise 明确说 \(10^4\) resamples，但 ordinary CHG CI 没有再明确：

* paired bootstrap；
* resample count；
* percentile / BCa；
* baseline 和 handoff 是否共享 indices。

由于“bootstrap intervals”是你们整个 audit 的 gate，建议至少补一句非常短的：

> **“All CHG intervals use a paired percentile bootstrap over evaluation examples with \(10^4\) resamples.”**

大概只占一行。

这比保留 permutation-test 之类不进入 gate 的信息更重要。

---

# 十五、Figure 3 caption 有一个轻微措辞不一致

Figure 3 现在写：

> “The identity control and the student sit in the top-left”

但图例实际展示的是：

> identity control (student suffix-native)

而 student self-prefill 的 full-context PPL 是一个不同的量，不应该直接进入同一 PPL 图。



以前你们已经很努力区分：

$$
PPL_{\rm identity,suffix}=21.2
$$

和：

$$
PPL_{\rm student,full}=25.4.
$$

所以建议 caption 改成：

> **“The identity control—the student's suffix-native reference—sits in the top-left.”**

不要再单独说 “and the student”。

很小的问题，但能避免 PPL 口径再次产生歧义。

---

# 十六、Conclusion 已经不错，但有一句仍可更稳

现在 Conclusion：

> “it is not explained by calibration volume, translator capacity, source-layer combination, or tokenization alignment”



这个表述略偏因果。

严格来说你证明的是：

> 在 tested range 中改变这些因素没有消除 deficit。

更科学的写法：

> **“the deficit persists across the tested calibration budgets, translator capacities, source-layer combinations, and token-alignment controls.”**

同一个意思，但不会让 Reviewer问：

> 你怎么证明未来更大 capacity 也解释不了？

---

# 十七、版面已经从 19 页压到 13 页，这是明显进步，但要看最终 venue

这版 References 从第 10 页中部开始。

所以真正 main text 大约是：

> **9 页多一点。**

如果最终 venue 的正文上限正好是 9 页，这一版仍然可能需要再压大约半页到 1 页。

最容易压的是：

* Section 6.4 的 context/cross-architecture/replay 合并；
* text-channel control；
* Section 7 两段压成一段；
* Limitations 的 MLP reproducibility 缩短后移 appendix。

但科研内容层面已经不需要再砍。

另外当前首页仍是实名、邮箱、CETC。

如果最终会议要求 double blind，记得投稿版匿名即可。

---

# 十八、我现在的正式评分

| 维度                       |      上版 |            最新版 |
| ------------------------ | ------: | -------------: |
| Novelty                  |     8.0 |        **8.0** |
| Technical Quality        |     8.5 |        **8.5** |
| Experimental Rigor       |     8.5 |        **9.0** |
| Statistical Rigor        | 7.5–8.0 |        **8.0** |
| Related Work             |     6.5 |        **9.0** |
| Claim–Evidence Alignment |     8.5 |        **9.0** |
| Reproducibility          |     9.0 |        **8.5** |
| Clarity                  |     8.5 |        **8.5** |
| Significance             |     8.5 |        **8.5** |
| **Overall**              | **6–7** | **7 / Accept** |

Reproducibility 为什么从 9 降到 8.5？

不是稿子退步，而是最新稿主动披露了：

> same-seed per-head MLP 仍不 deterministic。

这是一个真实存在但已经被诚实管理的问题。

---

# 十九、如果我是正式 Reviewer，我现在会给出这样的意见

> **Accept.**
>
> The paper has substantially matured. It no longer presents a universal negative claim about KV-cache translation; instead, it contributes a carefully scoped audit framework that separates injection mechanics, replacement-level fidelity, and incremental benefit from teacher-derived cache content. This distinction is timely given recent literature reporting outcomes ranging from severe degradation to near-native retention and gains above the receiving model's own baseline.
>
> I find two empirical observations particularly valuable. First, substantially improved held-out KV reconstruction does not improve downstream task performance: the larger joint MLP recovers student states better than the affine baseline but performs no better functionally. Second, near-native suffix perplexity does not imply recovered task capability. These results provide useful caution against common proxy metrics for cache-transfer quality.
>
> The revision also substantially improves the comparison with the closest published strict-zero-prefill baseline by implementing top-k source-layer concatenation, de-RoPE keys, and 1,024-token web-text calibration, while clearly disclosing the remaining calibration-volume and corpus differences.
>
> My remaining concerns are minor. The H1 “exactness” criterion should be stated with an explicit numerical tolerance rather than an equality to one; the proposed replacement/degradation labels should acknowledge that statistical degradation and practical non-inferiority are not mutually exclusive; and a few remaining causal or universal phrasings should be softened. These issues do not change the reported empirical results.
>
> **Score: 7/10 — Accept.**

---

## 二十、投稿前我只建议再做最后 8 个修改

1. **H1：** “exactly / =1 within float tolerance” 改成明确 numeric tolerance，最好增加 max-logit-error。
2. **Gate：** 把 replacement non-inferiority 与 significant degradation 当成两个统计维度，不要隐含 mutually exclusive。
3. **Abstract：** “four further scale pairs” 改成 **three additional / four total Qwen3 scale pairs**。
4. **Mapper list：** 解释 low-rank shared-basis 对应 Table 1 哪一行，否则删掉这个 family 名称。
5. **H3：** 标题改成 **No Teacher-Content Probe Establishes Gain**，Section 7 再弱化一句 probe-vs-translator 表述。
6. **Heo：** 不必再大规模实验；Conclusion 明确是匹配 concatenation + sequence length，但 calibration volume/domain 仍不同。
7. **Statistics：** 补一句 ordinary CHG CI 是 paired percentile bootstrap、\(10^4\) resamples。
8. **Copy/edit：** Figure 3 的 “identity control and student”、Conclusion 的 “not explained by”、MLP nondeterminism 原因说明等一次清理。

**做到这里以后，我建议停止继续增加实验。**

这版最重要的变化是：即使明天出现一个真正成功、能把 Teacher advantage 转给 Student 的 KV translator，**这篇论文仍然不会失效**。因为它的主要贡献已经从“translation 不成功”变成了：

> **怎样判断一个 cache-transfer system 到底只是 mechanics 正确、达到 receiver replacement，还是实际上获得了 teacher-derived capability gain；以及为什么 reconstruction 和 perplexity 都不能替代这个判断。**

这已经具备一篇成熟 audit/measurement 论文应有的独立价值。

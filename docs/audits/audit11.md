## **一、最新审稿结论**

审稿对象：《Cache Translation Across Model Scale: An Audit with Identity Controls and Teacher-Content Probes》，2026年11月22日版本，共15页。

本轮建议评分

# **6/10**

**Weak Accept / Borderline**

上一轮评分

# **7/10**

Accept

审稿信心

# **4/5**

较高

**我的总体判断是：论文已有足够的审计方法贡献，但11月新增相关工作带来了新的论证压力。**

相较9月版本，这次最值得肯定的是：新增了对近期研究的讨论，特别是功能导向的 KV 映射、知识蒸馏、选择性重计算以及跨模型推理架构。这些研究使论文的应用背景更完整。

不过，新的相关工作也改变了审稿标准：现在不仅需要证明传统重构式 Mapper 在当前条件下失败，更需要说明 **当其他研究已经报告功能保持或能力恢复时，本文的审计框架相比它们究竟提供了什么不可替代的证据。**

我仍然认可论文最重要的三个贡献：

1. **三层审计框架**：区分注入机制 H1、Prefill 替代 H2、Teacher 内容增益 H3。
2. **可复用的统计验证方法**：Identity Control、非劣效判据、Teacher-content Probe。
3. **两个有价值的实验发现**：KV 重构精度更高不等于任务能力更好；PPL 接近原生不等于能力恢复。

这些贡献没有因为近期出现新的成功方法而失效。

但本轮我不会直接维持7分，因为新稿的相关工作解释中出现了值得修正的逻辑问题，而且有几处上一轮指出的图文错误仍然保留。

## **二、最重要的问题：新增相关工作的解释存在逻辑冲突**

Major Concern 1｜优先级 P0

这是本轮我认为最需要认真修改的地方。

论文第9节前后的 Appendix E 把近期研究划分为若干其他方法族，包括 CacheLink、KV-Lingo、BridgeKV、RaReCache、ACCT 等，并试图得出一个统一解释：

> 这些方法之所以能够恢复能力，是因为增加了本文 strict zero re-prefill 不允许的计算，或者采用了接收端功能优化，因此并不改变本文的判断。

这个解释有一定道理，但**不能作为所有方法的统一结论**。

我查阅了相关工作的原始论文，发现至少三篇需要区别对待：

| 新工作       | 原论文的关键特点                                          | 对本文的影响                          |
| --------- | ------------------------------------------------- | ------------------------------- |
| CacheLink | 保持 zero-prefill，利用 frozen receiver 的功能反馈训练 K/V 修正 | 直接支持“重构精度不等于功能质量”，但也提出更强的功能优化方法 |
| KV-Lingo  | 通过蒸馏训练 Cache Translator，推理切换时避免重复 Prefill         | 说明严格零重 Prefill 与功能导向训练可以同时成立    |
| ACCT      | 利用源模型 KV 构造接收端稀疏 KV，无需重新编码原上下文                    | 说明部分成功结果仍属于 zero-prefill 相关设计空间 |

这些是各篇论文报告的方法和结果，并非我独立复现的结论。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arxiv.org+2



尤其 CacheLink 非常值得重视。它在严格 zero-prefill 下，用 receiver functional supervision 改善了任务表现，同时进一步验证 KV 重构误差与接收端实际行为之间的不一致。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arxiv.org+1



这与本文有很强的研究重叠，但并非完全相同：CacheLink 主要提出功能导向的映射方法，而本文主要提出审计流程和判据。

因此，**不能简单将所有近期成功工作解释为“额外重算使能力得以恢复”。** 蒸馏训练发生在离线阶段，并不自动违反推理阶段的 zero-re-prefill 约束。

### **建议直接重写 Appendix E 的核心结论**

建议替换为：

> *Recent work identifies several routes to successful cross-model cache reuse, including receiver-function-aware optimization, distillation-based translation, sparse KV transfer, and selective target-side recomputation. These approaches differ in whether they preserve strict zero re-prefill at inference. Our negative findings apply to the tested reconstruction-oriented and architecture-informed translators, not to all forms of functional cache transport. The proposed audit remains applicable to these newer methods by separately checking injection correctness, receiver replacement, and incremental capability gain.*

这个版本能够保留本文价值，而且不会与近期正面实验发生不必要的冲突。

与此同时，Section 7 的这句话：

> “Computation, not more cache, is what recovers the capability—which is our finding stated positively.”

也建议删除或改写。这是从外部新工作的结果推导出了一个本文实验尚未单独验证的因果解释。

**更好的论文定位应该是：本文指出重构目标的局限，近期工作则从功能优化等方向证明了其他可行路径。两者互补，而不是互相否定。**

## **三、创新性面临新的竞争：两项“解耦”发现不宜再作为独占创新**

Major Concern 2｜优先级 P0

原先本文非常有特点的两个发现是：

\\[ \text{KV reconstruction fidelity}\not\Rightarrow\text{task utility} \\]

\\[ \text{near-native perplexity}\not\Rightarrow\text{task preservation} \\]

这两个结果仍然有价值，而且本文给出了比较直观的数据：

- Joint MLP 的 Value \\(R^2\\) 达到 0.481，却未恢复任务表现。
- 某个翻译 Cache 的 suffix PPL 为 23.7，接近 identity 的 21.2，但 accuracy 仅为 0.267，而 Student 原生为 0.500。

问题在于，2026年10月的 CacheLink 已经独立研究了重构与实际功能之间的脱节，并通过 attention distribution、next-token distribution 等实验进一步解释这一现象。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arxiv.org



因此，截至11月投稿时，**“KV 重构精度不等于功能质量”已经不能再被表述成只有本文首次发现的结论。**

但这并不意味着本文失去创新性。

我建议把创新重点重新排序：

1. **首要贡献：** H1/H2/H3 三层审计框架与统计判据。
2. **第二贡献：** Identity Control 与 teacher-content intervention 作为评价工具。
3. **第三贡献：** 在相同审计体系下，揭示重构、生成流畅度与任务效果之间的不一致。

这样可以清晰区分：

- CacheLink 解决的是“怎样优化 KV 映射的功能效果”。
- 本文解决的是“怎样判断任何 KV 映射方法究竟达到了什么程度的成功”。

这才是论文在最新文献环境中最稳固的差异化定位。

## **四、最新稿仍然存在的具体问题**

### **1. Figure 4 的图文不一致仍未修正**

P0｜必须修改

第14页 Figure 4 的右图实际上展示的是 **8 个 layer octants**，横轴为 0–7，并存在两个正点估计：

\\[ +0.020,\quad +0.006 \\]

但 caption 仍然写：

> “Right: layer windows ... the top-third window is the only neutral one. These are the rows of Table 5.”

而 Table 5 展示的是 bottom/middle/top thirds，不包含 octant sweep。这个问题从上一稿延续至今。

建议直接把 Figure 4 caption 改为：

> *Figure 4: Teacher-content probes on the primary pair (n=30). Left: fraction probes under RAT and affine translators (Table 5). Right: exploratory affine octant sweep. No octant establishes statistically supported positive gain; the best point estimate is +0.020 [−0.012,+0.052]. The coarse layer-window results are reported separately in Table 5.*

这是纯粹的图文一致性问题，但现在投稿会显得不够严谨。

### **2. Abstract 有明显的重复文字**

P0｜必须修改

第一页 Abstract 出现：

> “and a 37M-parameter joint MLP does not help. help.”

其中第二个 `help.` 是重复文字。

main(20261122-065311).pdf



此外，Abstract 仍然写：

> “whether the injection pipeline consumes a foreign cache correctly”

但 H1 的 Identity Control 实际验证的是 **Student 自己的 Cache 是否能经过注入链路并恢复原生输出**，并不直接证明任意 foreign cache 的正确性。

建议改为：

> *whether the cache-injection pipeline faithfully preserves a valid student-space cache.*

这与第3节的 H1 正式定义完全一致。

### **3. 部分因果结论仍然超过实验支持范围**

P1｜应修改

第6.1节写：

> “Whatever the mappers recover does not appear to include the part of the cache the student needs in order to answer.”

该表述虽然增加了 *does not appear*，但仍然暗示恢复的信息缺乏任务相关内容。

实验实际证明的是：提高现有重构指标没有提升任务表现，并不能证明这些重构信息不包含任务相关信号。

建议统一改成：

> *The additional reconstruction fidelity achieved by these mappers is insufficient to restore the student's task behavior.*

另外，第14页 Table 4 的说明中称 Joint MLP 的 \\(R_V^2\\) “nearly doubles”，但数据是：

\\[ 0.323\rightarrow0.481 \\]

实际增加约49%，不是接近翻倍。建议改成 *improves value-channel&#x20;*\\(R^2\\)*&#x20;by approximately 49%*。

### **4. Appendix D 对 Heo 非线性实验的解释需要更加客观**

P1｜应修改

最新版新增 Appendix D，尝试解释为什么 Heo et al. 的非线性映射能够恢复部分性能，而本文的非线性模型没有成功。

这个方向正确，但当前文字说 Heo 的恢复结果来自：

> “a failure set selected after the fact”

原论文确实在六个模型对中的两个线性失败模型对上报告了最多 +37 个百分点的恢复，但仅凭这一信息，不能断定它属于事后选择导致的偏差。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arxiv.org



建议改成：

> *The nonlinear recovery in Heo et al. is reported for the two model pairs where its linear mapper degrades substantially, whereas our nonlinear results are evaluated over fixed held-out samples under a different calibration and scoring protocol.*

这里客观陈述差异即可，不要暗示对方统计处理有问题，除非进一步核实其具体实验协议。

### **5. H2 非劣效检验的样本量论证应加条件**

P1｜统计精修

第3.2节新增：

> 当 n=30、区间半宽约 0.17 时，\\(\epsilon=0.02\\) 的非劣效检验大约需要 \\(2\times10^3\\) 个样本。

这个估算在假设效应均值接近零、方差维持不变的情况下有一定依据，但不能作为通用结论。

例如当前最佳强 Student 配置：

\\[ \widehat{\Delta}=-0.138 \\]

如果真实效应确实接近 -0.138，单纯扩大样本量并不会让它通过 \\(-0.02\\) 的非劣效门槛。

建议增加一句：

> *This rough sample-size estimate assumes a true effect near zero and variance comparable to the observed runs; increasing sample size alone cannot establish non-inferiority for a genuinely inferior mapper.*

这会让 H2 的统计解释更加严谨。

main(20261122-065311).pdf



## **五、是否还需要增加实验？**

我不建议继续增加普通 ridge、affine、MLP 或者更多 calibration budget。

当前已经有133个 translated configurations、48个 probes，实验设计范围相当丰富。

main(20261122-065311).pdf



真正值得考虑的是增加一个**正向验证实验**。

过去论文主要证明“哪些方法没有通过审计”，但对“当映射确实可恢复时，审计体系能否准确识别成功”展示得还不充分。

我建议优先考虑前面讨论过的固定 V-permutation 实验：使用 Student 原生 Cache，保持 K 不变，对 V 施加跨样本固定的随机维度排列，使用相同 ridge 管线学习逆映射，在独立测试集比较 Native、Permuted、Recovered 三种结果。

预期结果是：

\\[ \text{Native}\approx\text{Recovered}>\text{Permuted} \\]

如果成立，就能证明映射实现并非连已知可逆线性变化都无法学习。这将构成一个 **Synthetic Recoverability Positive Control**，与 H1 的 Identity Control 形成互补。

如果资源更充分，学术价值更高的是把 KV-Lingo 或 CacheLink 的功能导向 Mapper 放进同一 H1/H2/H3 审计体系。不过这会涉及方法复现、训练预算和任务协议对齐，不宜作为本轮投稿必须完成的工作。

## **六、当前论文的投稿风险**

还有两个提交层面的问题需要明确。

首先，最新版正文进入第10页才结束，References 从第10页开始。如果仍针对 ICLR 2027 的格式，官方规定投稿主文不超过9页、且必须双盲匿名。当前 PDF 首页仍包含作者姓名与单位。

[image](https://www.google.com/s2/favicons?domain=https://iclr.cc\&sz=32)

iclr.cc



其次，**ICLR 2027 的正式论文截止日为2026年9月25日，讨论期修改截至11月18日**。现在是2026年11月22日，所以这份稿件不能作为普通新投稿提交该届会议，也已超过常规讨论期的修改窗口。若论文已经处于该会议审稿流程，应遵循 OpenReview 对现有稿件的规定；否则需要选择下一投稿窗口。

[image](https://www.google.com/s2/favicons?domain=https://iclr.cc\&sz=32)

iclr.cc



此外，论文多处声称 release runs、protocol 和 instruments，但当前 PDF 没有清楚给出可访问的数据或代码入口。正式提交前应提供可审计的匿名仓库、版本标识及运行说明，至少保证 Reviewer 能找到这些材料。

## **七、综合评分**

| 审稿维度         | 9月版本     | 11月最新版   | 判断                           |
| ------------ | -------- | -------- | ---------------------------- |
| 创新性 Novelty  | 8.0      | 7.0      | 新工作与部分实证贡献重叠                 |
| 技术质量         | 8.5      | 8.0      | 主体可靠，新文献解释需要修改               |
| 实验严谨性        | 9.0      | 8.5      | 原有实验扎实，但缺正向审计示例              |
| 统计严谨性        | 9.0      | 8.0      | 基本良好，非劣效与样本量解释仍可优化           |
| Related Work | 9.0      | 7.0      | 覆盖更全，但新工作的技术边界解释不够准确         |
| 可复现性         | 9.0      | 8.5      | 运行记录完整，公开可用性仍需确认             |
| 论文清晰度        | 9.0      | 7.5      | Appendix E、图文和个别表述影响一致性      |
| 研究价值         | 8.5      | 8.0      | 审计方法仍有长期价值                   |
| **Overall**  | **7/10** | **6/10** | **Weak Accept / Borderline** |

这个分数下降不是因为之前的实验被推翻了，而是因为**11月的知识背景发生了变化，论文需要更严格地说明与近期成功方法的边界和区别**。

## **八、正式审稿意见**

> **Recommendation: Weak Accept (6/10).**
>
> This paper presents a useful methodological audit of cross-model KV-cache translation, separating injection mechanics, receiver-prefill replacement, and incremental benefit from teacher-derived content. The identity controls, statistical gates, and empirical observations concerning reconstruction–utility and perplexity–accuracy decoupling remain valuable.
>
> The manuscript is strengthened by its inclusion of recent concurrent work. However, several of these newer methods operate under zero-prefill inference while optimizing receiver functionality rather than cache reconstruction. The current discussion occasionally interprets their positive results as evidence that additional target-side computation is required, which is not generally supported.
>
> The novelty of the reconstruction–utility decoupling also needs to be positioned more carefully relative to CacheLink. I recommend emphasizing the three-stage audit protocol and its independent statistical criteria as the primary contribution, while presenting the decoupling observations as supporting empirical evidence.
>
> Several presentation inconsistencies remain, particularly the Figure 4 caption, the duplicate word in the abstract, and a few overly broad causal interpretations. These are correctable without changing the core experimental findings.
>
> **Overall, I view the paper as a promising measurement contribution that requires a focused revision of its related-work interpretation and claim boundaries before I would recommend a confident Accept.**

## **九、最终修改建议**

我建议这次只做一次有针对性的修改，优先级如下：

| 优先级 | 修改事项                                             | 工作量 |
| --- | ------------------------------------------------ | --- |
| P0  | 重写 Appendix E，准确区分 zero-prefill 功能优化、蒸馏映射和选择性重计算 | 中   |
| P0  | 将三层审计框架明确作为论文第一贡献，重新定位两项解耦结果                     | 小   |
| P0  | 修正 Figure 4 图文不一致和 Abstract 重复文字                 | 小   |
| P1  | 优化 Appendix D 对 Heo 非线性实验的描述                     | 小   |
| P1  | 降级 §6.1 和 §7 中的因果推论，完善非劣效样本量说明                   | 小   |
| P1  | 补足代码与数据可获得性说明，完成最终图表一致性检查                        | 小   |
| 可选  | 增加一次 V-permutation 正对照实验                         | 小至中 |

**最后的判断：这篇论文没有必要再回到早期“证明跨模型 KV Cache 能力不可迁移”的思路。**

相反，随着 KV-Lingo、CacheLink、BridgeKV、ACCT 等工作出现，更合理的论文主张应该是：

> **跨模型 KV Cache 迁移已经有多种技术路线和成功标准；本文提供一套独立于具体 Mapper 的审计方法，明确区分注入正确性、接收端 Prefill 替代和真正的能力增益。**

把这个主张贯彻到 Abstract、Introduction、Related Work、Appendix E 和 Conclusion 后，我认为它能够重新稳定在 **7/10 Accept** 的水平，而且论文的研究定位会比两个月前更加清晰。
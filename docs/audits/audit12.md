## **一、最新复审结论**

审稿对象：2026年11月22日修订稿《Cache Translation Across Model Scale: An Audit with Identity Controls and Teacher-Content Probes》，15页。

本轮总体评分

# **7/10**

**Accept**

上一轮评分

# **6/10**

Weak Accept / Borderline

审稿信心

# **4/5**

较高

**这次修改是有效的。我愿意将评分从上一轮的6/10恢复至7/10。**

最主要的原因不是又增加了实验，而是论文终于较好地处理了2026年9—11月出现的新研究成果，进一步明确了自身的创新边界。

现在论文的定位更加准确：

> **不是证明跨模型 KV Cache 能力迁移不可行，而是建立一套与具体映射算法相对独立的审计体系，分别验证注入正确性、Prefill 替代效果以及真实能力增益。**

目前没有发现需要推翻 H1/H2/H3 框架的核心缺陷。不过，独立复审后，我仍发现几处值得在投稿前修改的技术与统计细节，其中一处涉及重构实验的数据对应关系，需要核实。

## **二、上一轮审稿意见落实情况**

| 上轮问题                                         | 最新版情况                                 | 复审结论      |
| -------------------------------------------- | ------------------------------------- | --------- |
| Abstract 重复 `help.`                          | 已删除                                   | 已解决       |
| H1 将 Identity Control 误写成任意 foreign cache 验证 | Abstract 已改为有效 Student-space Cache 保真 | 基本解决      |
| 将两个解耦发现当作独占创新                                | 已改成 supporting evidence，并承认并行工作       | 已解决       |
| Appendix E 把成功归因于额外 Prefill 计算               | 已区分功能优化、离线蒸馏、选择性重计算和稀疏迁移              | 已解决       |
| Appendix D 对 Heo 非线性实验的解释不够客观                | 已用实验设置与指标差异进行比较                       | 已解决       |
| Figure 4 的 octant 图与 Table 5 对应错误            | 已明确右图是探索性八分层扫描                        | 已解决       |
| Joint MLP 的 \\(R^2\\) “接近翻倍”                 | 已正确改为提升约49%                           | 已解决       |
| 非劣效样本量估计没有前提                                 | 已补充真实效应接近零及方差稳定的条件                    | 基本解决      |
| 审计材料可获得性说明不足                                 | 已增加匿名归档及复算脚本说明                        | 改善，仍需交付验证 |

这几项修改可以在最新稿的 Abstract、§3.2、§6.1、Appendix B、D、E 中找到。

其中我认为最重要的是 **Appendix E 的重写**。

作者现在明确承认，CacheLink、KV-Lingo 等方法可以在推理时不重新 Prefill 的条件下，通过功能导向训练改善映射结果。论文不再把所有成功都解释为额外在线计算，并将自己的负结果限制在测试过的重构导向及架构辅助映射器中。

这与近期原论文的描述一致：CacheLink 直接针对接收端 attention 和预测行为优化；KV-Lingo 则通过离线蒸馏训练轻量映射器，并在切换时避免重新处理上下文。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arxiv.org+1



因此，**上一轮最大的文献定位问题已经得到实质解决。**

## **三、独立复审发现的剩余问题**

### **问题1：重构精度与任务表现是否来自同一训练实例？**

P1｜最值得核查的技术问题

这是这次审稿中新发现、比普通文字问题更重要的一点。

第4节写明，held-out KV reconstruction 的评估使用：

- 20个 calibration contexts；
- 10个独立 held-out contexts。

但 Appendix Table 4 把 Joint MLP 标记为 `c=30`，又将其 \\(R_V^2=0.481\\) 与 `c=30` 的下游任务 CHG（−0.30 至 −0.26）直接关联。

main(20261122-071334).pdf



这里需要确认：

**R² 和 CHG 是否来自同一个训练好的 Mapper checkpoint？**

如果 R² 使用20个上下文拟合的模型，而 CHG 使用另外一个30上下文拟合的模型，就不能直接以这两个结果证明“同一个映射器重构提高但功能未提高”。

建议在 Appendix Table 4 下增加说明，明确 checkpoint ID、训练数据划分以及 R² 和 CHG 的对应关系。

如果数据确实来自同一 checkpoint，只需要把流程交代清楚，这个问题就能关闭。如果不是，我建议重新使用同一 checkpoint 计算这两个指标。

### **问题2：最大的正点估计存在统计范围歧义**

P1｜必须修改

第5.2节仍然写：

> “The largest point estimate anywhere in the audit is +0.010.”

但 Appendix C 的 octant sweep 实际有：

\\[ +0.020\\;[-0.012,+0.052]. \\]

因此，如果 *anywhere in the audit* 包括全部 Probe，这句话并不成立。

main(20261122-071334).pdf



建议直接改成：

> *The largest point estimate among translated-cache configurations is +0.010; the exploratory octant probes include a larger positive point estimate, but none establishes significant gain.*

这不会改变任何结论，只是明确 Translation 与 Probe 的统计范围。

### **问题3：复现说明仍有一处内部矛盾**

P1｜应修改

Appendix B 新增了匿名归档说明，这是好事。

但是正文写：

> “Table 1 and Tables 5–4 are recomputed from those stored scores alone.”

这里有两个问题：

第一，`Tables 5–4` 顺序颠倒，应明确写出表号。

第二，Table 4 报告 KV reconstruction \\(R^2\\)。如果归档只包含每个样本的最终回答分数，那么无法从这些分数重新计算 KV \\(R^2\\)。

建议把可复现性分成两类：

- **任务指标与 CI**：从 per-sample scores 和 bootstrap script 直接复算。
- **KV reconstruction 指标**：需要保存相应的 KV 状态、重构预测值或足够的统计数据，并给出计算脚本。

目前 PDF 声称归档随投稿一并提供，但本次只上传了论文 PDF，因此我无法独立验证归档是否完整。

### **问题4：少数表述仍然不够严谨**

P2｜文字与逻辑精修

我建议一次性调整下面几处：

| 位置      | 当前表述或问题                                                    | 建议                                                               |
| ------- | ---------------------------------------------------------- | ---------------------------------------------------------------- |
| §3.1    | “a student with no signal sits at 0.25”                    | 改成“uniform four-choice predictions yield 0.25”；无任务信号不必然意味着模型输出均匀 |
| §3.2    | “no increase in sample size can establish non-inferiority” | 增加“if the true effect remains below −ε”，明确是条件性结论                 |
| §3.2 H3 | “The probe does not train a translator”                    | 改成“the probe adds no translator training”；因为它依赖已拟合的 RAT/affine   |
| Table 1 | “Teacher full prefill, 1.7B”                               | 建议明确标成“Teacher Qwen3-4B full-prefill”，避免与 Student 1.7B 混淆        |
| §8      | “Four families outside it are catalogued”                  | Appendix E 实际讨论了多种路线，建议用“several alternative families”避免分类计数歧义   |

这些属于小修，不会影响论文评分。

## **四、论文现在的创新性是否足够？**

我认为**足够支撑一篇方法学论文**，但不宜将其理解成强理论突破。

当前创新定位

# **H1**

Mechanics

注入链路验证

# **H2**

Replacement

非劣效替代验证

# **H3**

Gain

Teacher 内容增益验证

核心贡献不是新的 KV Translator，而是一套分层、可审计的验证协议。

最重要的是，近期文献恰好增强了这一框架的必要性。

例如，KV-Lingo 使用蒸馏而不是单纯 KV 重构；CacheLink 直接优化接收端行为；BridgeKV 结合学习式 KV 对齐与下游预测约束；ACCT 使用查询相关的稀疏 KV 状态。它们在训练目标、Cache 结构和任务协议上不同，因此更需要共同的成功判据。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arxiv.org+3



但本文的审计框架目前主要在失败案例上展开。严格来说，它还没有通过一个完整的正向方法复现来证明三层 Gate 在成功案例中的判别效果。

因此，**框架的合理性已经较强，跨方法普适性的实证验证仍然有限。**

这也是我认为目前7分而不是8分的主要原因。

## **五、最新评分**

| 维度                       | 上一轮      | 最新稿                    |
| ------------------------ | -------- | ---------------------- |
| Novelty                  | 7.0      | **7.5**                |
| Technical Quality        | 8.0      | **8.0**                |
| Experimental Rigor       | 8.5      | **8.5**                |
| Statistical Rigor        | 8.0      | **8.5**                |
| Related Work             | 7.0      | **8.5**                |
| Claim–Evidence Alignment | 7.5      | **8.5**                |
| Reproducibility          | 8.5      | **8.0**，待核查 R² 对应关系和归档 |
| Clarity                  | 7.5      | **8.5**                |
| Significance             | 8.0      | **8.0**                |
| **Overall**              | **6/10** | **7/10 — Accept**      |

这里的7分是我作为独立 Reviewer 的建议评分，不代表会议的实际接收概率；顶会的录用还取决于其他审稿人、论文竞争情况和投稿规则。

## **六、正式 Reviewer 结论**

> **Recommendation: Accept (7/10).**
>
> The revision addresses the principal concerns from the previous review. The authors now position the three-stage audit protocol as the primary contribution, distinguish their empirical decoupling observations from concurrent findings, and accurately characterize alternative functional cache-transport approaches that preserve zero-prefill inference.
>
> The revised discussion of nonlinear recovery in prior work is appropriately cautious, and the corrected figure captions and statistical qualifications improve the manuscript's internal consistency.
>
> I consider the audit framework a useful methodological contribution. Its separation of injection mechanics, receiver replacement, and incremental teacher-derived gain provides a common evaluation structure across different translator families.
>
> My remaining concerns are limited. The reconstruction diagnostic should explicitly establish that its reported \\(R^2\\) and downstream outcomes correspond to the same trained mapper instances. The scope of the maximum positive point-estimate statement should be corrected, and the reproducibility appendix should distinguish metrics reconstructible from answer scores from those requiring saved KV representations.
>
> These issues do not invalidate the central findings, but addressing them would improve the paper's evidential transparency.
>
> **Score: 7/10 — Accept. Confidence: 4/5.**

## **七、最终建议**

我建议**不再扩展研究方向，也不必继续增加普通 Mapper 实验**。当前优先完成三件事：核实 R² 与 CHG 的 checkpoint 对应关系，修正最大点估计的统计范围，以及检查复现归档是否真的覆盖所有指标。

这三项解决后，就可以冻结科学内容。若投稿双盲会议，再制作符合相应页数限制的匿名版本；当前稿件首页仍有作者信息。

这篇论文经过两个月的演化，已经形成了相对稳定且有独立价值的研究贡献：

**从“证明某些 KV Cache 映射器失败”，转变为“提出判断跨模型 Cache 迁移究竟是否成功的统一审计方法”。**

我认为这是正确的方向，也足以成为这篇论文最终的核心主张。
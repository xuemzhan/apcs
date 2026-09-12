# 审稿意见：Cache Translation Across Heterogeneous LLMs

## 总体评价

这是一篇方法论扎实的**审计型（audit）论文**，核心贡献不是提出新的更强的跨模型 cache 迁移方法，而是设计了一套三假设分级框架（H1 机制 / H2 映射 / H3 可利用性）加上两个此前文献缺失的关键工具——**身份对照（identity control）**和**oracle 探针**——用来拆解"cache translation 有效"这类主张究竟成立在哪一层。整体逻辑链条清晰、统计处理规范（bootstrap CI、置换检验、disjoint calibration/eval split），并且诚实报告了负面结果和自身实现中的 bug（如 prompt 渲染两次导致置信度虚高）。这种"审计清单"式的贡献对领域是有价值的。

但从投稿角度看，论文目前的**实证范围与其表述的普适性主张之间存在明显落差**，这是审稿人最可能提出的核心质疑。以下按主次列出问题。

---

## 主要贡献（认可点）

1. **身份对照（H1）**——把学生自己的 cache 过一遍完整注入管线，隔离"机制损耗"和"映射损耗"。这个对照组极其简单却是此前所有相关工作都没做的，是本文最有说服力的设计。
2. **Oracle 探针（H3）**——不训练任何东西，直接把 teacher 内容按比例/按层窗口混入 student 自己的 cache，从而给出"任何 translator 理论上限"的上界估计。这是把 H2 的失败和"是不是我们的映射器不够好"这个反驳解耦的关键一步，逻辑上很漂亮。
3. **PPL 与 accuracy 解耦的发现（5.3, 5.5, Figure 4）**——一个 PPL 修复到接近原生水平（23.7 vs 21.2）的 cache 仍然只有 0.267 的准确率，这直接打掉了"perplexity 可作为迁移质量代理指标"这一在 serving 文献中常见的隐含假设，是一个干净且可推广的方法论警示。
4. 承认并修正了自己早期协议中的 bug（选项渲染两次导致置信度虚高），这种透明度应该被鼓励。

---

## 主要问题（Major Concerns）

**M1. 实证范围与标题/摘要的普适性表述不匹配。**
标题是"Across Heterogeneous LLMs"，摘要和结论用了"the teacher's answer-relevant advantage does not survive KV-space translation"这种近乎全称的表述，但实际只测了**一个模型家族（Qwen3）、两对模型、两个英文多选任务、n=30–63**。Limitations 部分坦诚承认了这一点，但摘要和结论没有相应收窄措辞。建议要么扩大到至少一个跨家族的 pair（如 Qwen→Llama），要么把标题/摘要的claim范围明确限定为"within-family, MCQA setting"。目前这个落差是最容易被拒稿的理由。

**M2. 21次GPU运行对Table 2里的配置数量而言样本量偏薄，且CI只覆盖了评估样本的抽样不确定性，没有覆盖映射器训练的随机性。**
每个配置似乎是单一硬件、单一种子跑一次（8节提到"multi-seed sweep in appendix-scale runs showed the same signs"，但正文没有展示这些数据）。ridge/affine在30-200个校准样本上训练，这类小样本估计器对种子/校准子集的方差可能不小，而目前报告的95% CI只是对评估集做bootstrap，并不能反映"如果重新训练一次映射器结果会有多大波动"。这一点需要要么补充多种子的映射器训练方差，要么在正文（不只是附录提及）里明确说明CI的覆盖范围仅限于评估阶段。

**M3. 对已发表工作（MoT、C2C、Heo et al. ridge mapper）的批评是分析性的，而非直接复现对比的。**
论文的核心论证之一是"MoT 之类工作报告的高准确率来自 target-side replay/correction，不是纯粹的 cache 迁移"，但8节明确写"We did not re-implement MoT's full replay pipeline"。这意味着这个关键的反驳性论断（"别人的正面结果是因为掺了别的东西"）目前是**推断**而非**实测**。对于一篇以"audit"为卖点的论文，这里恰恰应该有一个受控对比：至少复现MoT或Heo et al.其中一个的完整设置，套上你们的identity control和oracle probe，直接展示"加上我们的对照后，他们报告的正面结果会缩水多少"。目前没有这个实验，论点停留在架构层面的合理推测，说服力打折扣。类似地，你们自己的ridge-per-head（−0.296）比Heo et al.报告的73–98%留存率差很多，这个差异是校准协议不同、pair不同，还是实现细节不同，正文没有直接讨论，是一个明显的"审稿人会问"的空白。

**M4. RAT（residual-anchored translator）作为一个重点方法论贡献，其消融结果实际上削弱了它自身的必要性。**
5.3节诚实地报告了"去掉analytic core、提高correction rank反而修复了fluency"，且RAT在Table 2里的表现并不优于更简单的affine mapping（甚至更差：−0.268 vs affine的−0.138）。这本身是很好的科学诚实，但也说明RAT作为"贡献2"的分量不够——如果最终结论是"架构感知的复杂设计不如简单的per-head affine"，那么RAT在论文结构中的定位需要调整：它更像是"我们也验证了一个更复杂的假设，同样失败了"的补充证据，而不是与ridge/affine并列的一个主推方法。目前的呈现方式容易让审稿人觉得作者对自己重点设计的方法投入不够批判性反思。

**M5. "text channel"对比（teacher摘要 0.413 vs student full-text 0.508）在Implications部分突然出现，缺少对应的方法学描述。**
这是一个很有价值的对比（证明"文本压缩"比"cache迁移"更划算），但读者不知道这个摘要是怎么生成的、用什么prompt、是否也做了disjoint eval、是否有CI。需要在4节或新增一小节里补充这个实验的设置细节，否则它读起来像是临时加的一句话结论，权重与其重要性不匹配。

---

## 次要问题（Minor）

- Table 1 中"This paper"一行写"none: identity control + oracle bounds"作为"unexamined assumption"，语义上略怪（应该是"我们没有未检验的假设"的意思），建议改写更清楚。
- Figure 1 的pipeline图里"H2 fail ⇒ probe ceiling"这个箭头逻辑值得在正文里更明确地重述一遍：为什么H2失败要触发H3而不是直接终止审计？（论文里其实有很好的理由——H3能在不训练任何东西的情况下给出上界——但这个"为什么H3是决定性的"目前只在4.2节末尾一句带过，建议提到引言或框架小节里更醒目地强调。）
- Channel asymmetry（5.6节，key-only vs value-only）作为一个"finding"被报告但"leave the mechanism open"——如果篇幅允许，哪怕是一个简单的假设性解释（比如key决定attention routing、value决定content）也会让这部分更完整，目前有点像未完成的实验挂在那里。
- 结论句"The teacher's capability is in its weights, not in its cache"作为标题级金句很抓人，但严格说这是从一个family、两个task外推出来的强论断，建议在abstract和conclusion里都加一个scope qualifier（如"within the tested family/regime"），和Limitations保持一致的谦逊度。
- 我注意到这版论文的任务集是HellaSwag/ARC-Challenge（英文多选），如果这和你原计划里"以中文长文档QA为主任务"的方向不一致，值得确认一下这是刻意的阶段性版本（先用标准英文benchmark做方法验证，中文长文档QA留给后续）还是需要在这版里补充说明任务选择的理由——审稿人大概率会问"为什么选这两个任务，是否能代表你声称的应用场景（长上下文prefill复用）"，多选题本身其实不太需要长上下文推理能力，这和cache translation要解决的"长上下文prefill复用"这个motivation之间有一点点脱节，值得在方法或限制部分补一句解释任务选择与应用场景的关系。

---

## 建议向作者提出的问题

1. 能否补充至少一个跨模型家族的pair（如Qwen→Llama或Qwen→Gemma），以支撑"heterogeneous"这个标题用词？
2. 能否展示"appendix-scale multi-seed sweep"的具体数据，而不只是文字提及"showed the same signs"？
3. Ridge-per-head的−0.296和Heo et al.报告的73–98%留存率之间的巨大差异，具体来自协议差异的哪一部分？
4. Text channel对比实验的具体设置（prompt、样本量、CI）能否补充？
5. HellaSwag/ARC-Challenge这类短上下文多选任务，与论文motivation中强调的"长上下文prefill复用"场景之间的关联性如何论证？

---

## 审稿结论

**建议：Weak Accept / Major Revision（视会议标准，接近ICLR的"6: marginally above threshold"）**

论文的方法论设计（identity control + oracle probe + 分级假设）本身是一个真正有增量价值的贡献，负面结果的报告方式诚实且统计处理规范，这在这个方向的文献里是稀缺的。但目前的实证范围（单一模型家族、两个短上下文英文多选任务、单种子）不足以支撑标题和摘要中的普适性措辞，且对已发表工作的批评（MoT等的正面结果"掺水"）目前停留在分析层面而非直接复现对比，这是审稿人最容易抓住的软肋。

如果作者能够：(a) 收窄摘要/标题的claim范围以匹配实际实验范围，或者补充跨家族实验；(b) 至少对一个竞品做受控复现对比；(c) 补充text channel实验的方法细节，这篇论文可以达到清晰的accept水平。目前版本建议按major revision处理。
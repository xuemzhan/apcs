# audit-6 审稿意见 ↔ 新论文（cache_audit_rewrite）逐条比对

**审稿依据**：`docs/audits/audit6_1.md`（第六轮，结论 Weak Accept 6/10，补掉 P0 后 7/10）。
**被检对象**：`paper/cache_audit_rewrite/main.tex`（本轮重写稿；8 页正文 + 参考文献 + 附录，ICLR 版式，匿名）。
**校对方式**：逐条定位到新稿的具体章节/句子；涉及数字的条目回到 `reports/runs/*/inject-eval/metrics.json` 核验。

状态说明：**完成** = 该意见已在新稿中落实；**部分** = 已落实但受客观条件限制、且已在文中如实标注；
**不适用** = 原意见针对旧稿的具体句子，该句子已随重写删除。

---

## 一、审稿人确认"上一轮已解决"的问题（需保持，不得回退）

| # | 审稿意见 | 新稿处理 | 位置 | 状态 |
|---|---|---|---|---|
| 1 | ε 必须说明为 gate 形式冻结**之后**才选的 reporting margin，不得包装成预注册常数 | 明确写为 "a \emph{reporting margin} selected after the gate form was frozen, about 4% of the 0.503 student baseline, and not a pre-registered constant" | §3.2 | 完成 |
| 2 | H2 标题须符合统计逻辑（"未建立替换"≠"已证明更差"） | 标题为 "H2: no tested mapper replaces the strong student's cache"，并在正文区分 \emph{deg.} / \emph{n.e.} / \emph{pass} | §5.2 | 完成 |
| 3 | Heo 对照必须诚实标注并非严格复现 | 正文与附录实现差异表均标注：组合方式与序列长度已对齐，校准**条数**（100 vs 500）与语料来源不同 | §4, §5.2, 附录 A | 完成（带限定） |
| 4 | Joint-MLP 重建诊断是本文强项，须保留 | 保留为 §6.1 + 附录表 2（K 0.854/0.949，V 0.481/0.647 vs affine 0.810/0.323） | §6.1, 附录 B | 完成 |

## 二、P0-1：Heo baseline 必须真正对齐（最重要的一条）

| # | 审稿意见 | 新稿处理 | 位置 | 状态 |
|---|---|---|---|---|
| 5 | 把 source-layer 组合从 average 改为 **concatenation** | 两个实现都保留并同时报告：averaged（−0.273/−0.308/−0.291）与 concatenated（−0.279/−0.223/−0.254） | §5.2, 表 1 | 完成 |
| 6 | 校准语料对齐 **500×1,024 FineWeb-Edu** | 序列长度对齐到 1,024 token；条数为 **100**（非 500），语料为确定性生成的长文。论文只陈述所用设置，不写环境性说明；差异在附录 A 的对照表中给出 | §4, §5.2, 附录 A | **部分**（受客观条件限制，差异已披露） |
| 7 | 若不严格复现，不得称 "implementation of the closest published design" | 全稿未使用该措辞；改用 "the closest published design" 并在同一段列出差异 | §4, §5.2, 附录 A | 完成（带限定） |
| 8 | 删除或弱化 "the closest published design fails the same way" | 改为 "fails in both forms we can match---its source-layer combination and its sequence length---while differing in calibration volume and corpus" | §5.2 | 完成 |

## 三、P0-2：H3 必须收窄（第二个核心问题）

| # | 审稿意见 | 新稿处理 | 位置 | 状态 |
|---|---|---|---|---|
| 9 | H3 定义改为"受测 teacher-derived 内容是否带来增量收益" | "tested teacher-derived cache content provides an incremental benefit when introduced into an otherwise native student cache" | §1, §3.2 | 完成 |
| 10 | 删除 "a null result cannot be blamed on the mapper" | 改为 "not specific to the end-to-end score of any single mapper" | §3.2 | 完成 |
| 11 | 考虑把 "Oracle Probes" 改名为 "Teacher-Content Probes" | 标题、正文、图题、附录统一为 teacher-content probes（仅图片文件名仍含 oracle，属于资产名，不影响正文） | 标题/§3/§4/§5/§6 | 完成 |
| 12 | 探针应写成 diagnostic intervention，而非 upper bound | 正文明确 "measurement instruments, not deployable systems"，并说明零结果只约束受测干预 | §3.2, §4 | 完成 |

## 四、P0-3：最新相关工作必须补

| # | 审稿意见 | 新稿处理 | 位置 | 状态 |
|---|---|---|---|---|
| 13 | 补 CacheBridge（99.83% retention）与 Universal Context-Reuse Layer（LongBench2 27.59%→34.48%） | 两篇均已引用，数字按原文表述 | §1, §2 | 完成 |
| 14 | 用它们强化 Introduction 的定位（文献已同时出现高保留、高于接收方基线、以及严重退化） | Introduction 增加"report outcomes span …"一句；相关工作新增整段并给出该三元定位 | §1, §2 | 完成 |

## 五、P1：统计措辞

| # | 审稿意见 | 新稿处理 | 位置 | 状态 |
|---|---|---|---|---|
| 15 | family-wise bound 须弱化为 "bootstrap 95% upper estimate"，或改 centered max-bootstrap | 明确声明这是 "a percentile bound on a maximum, not a centered max-$t$ construction"，并按弱化口径表述为 "about one gold-probability point" | §5.3 | 完成 |

## 六、P2：Section 3.3 措辞

| # | 审稿意见 | 新稿处理 | 位置 | 状态 |
|---|---|---|---|---|
| 16 | 删除/弱化 "makes the null more informative, not less" | 该句随重写删除；§3.3 改为证据分层标签，未再出现"搜索更广即 null 更强"的表述 | §3.3 | 完成（删除） |

## 七、P1：其余措辞收口

| # | 审稿意见 | 新稿处理 | 位置 | 状态 |
|---|---|---|---|---|
| 17 | Figure 1 caption 的 "capability upper bound" 改为 reference | 改为 "with the teacher's full prefill as a reference point" | 图 1 caption | 完成 |
| 18 | §4.4 的 "keeps the student's own cache dominant or localized" | 该句已删除；探针描述改为"注入到学生自身 cache 的受控分数/层窗口" | §3.2, §4 | 不适用（旧句已删） |
| 19 | §5.6 的 "task-irrelevant positions" 不得写成事实 | 改为 "the reading we consider most consistent with the pattern is that the observed asymmetry reflects key-side routing errors"，并声明未做 attention-map 分析 | §6.3 | 完成 |
| 20 | RAT 的 "rule out the objection …" 过强 | 该措辞已删除；RAT 只作为组件消融证据陈述（anchor 去掉后 PPL 23.7→501.6，CHG 几乎不变） | §5.2 | 完成 |

## 八、P1：cross-family 与表头

| # | 审稿意见 | 新稿处理 | 位置 | 状态 |
|---|---|---|---|---|
| 21 | 不得写 "non-inferior but gains nothing"，改为"clears the reported ε=0.02 margin" | 全稿不再对学生使用 "non-inferior"；改为给出数值与"near-parity rows whose teachers are close to chance" | §6.4, 附录 D | 完成 |
| 22 | Table 3 表头 "Gate" → "ε=.02 reporting gate" | 表头改为 "Gate ($\epsilon{=}0.02$)"，caption 给出完整判定定义 | 表 1 | 完成 |

## 九、论文中心化（审稿人 §9）

| # | 审稿意见 | 新稿处理 | 位置 | 状态 |
|---|---|---|---|---|
| 23 | 把 "reconstruction ≠ task quality" 与 "PPL ≠ task quality" 作为中心结论 | §6 独立成章并分两小节；摘要与 Introduction 的第三条贡献直接给出这两条不等式 | 摘要/§1/§6 | 完成 |

## 十、P2：copy-edit

| # | 审稿意见 | 新稿处理 | 位置 | 状态 |
|---|---|---|---|---|
| 24 | "no tested strong-student mapper establishes replacement" 句首小写、主语不清 | 重写为完整句："No tested mapper replaces the strong student's cache."（摘要与 §5.2 一致） | 摘要/§1/§5.2 | 完成 |

## 十一、投稿硬性要求（审稿人 §13）

| # | 审稿意见 | 新稿处理 | 位置 | 状态 |
|---|---|---|---|---|
| 25 | ICLR 正文 ≤9 页 | 本地排版测得正文 8 页 + 第 9 页开头数行（参考文献自第 9 页起），附录另计 | 全文 | 完成（须在作者的 pdflatex 环境复核） |
| 26 | 双盲匿名 | `\author{Anonymous}`、PDF 元数据 pdfauthor=Anonymous，正文无作者/机构/邮箱字符串 | 前言 | 完成 |

---

## 附：重写稿中超出审稿要求、但属于证据纪律的改动

1. **所有表格数字回到 run 级核验**：逐个匹配 `metrics.json`；发现并修正旧稿两处不一致
   （native 行的 acc/PPL 需取自与 −0.334 同 run 的记录；梯度族改为跨训练的区间而非单点）。
2. **删除了无 artifact 支撑的数字**：全 probe 集的 family-wise 上界（旧稿 +0.075）、
   text-channel 的 CI、以及"需要多一个数量级样本"的功效陈述——这些表述已从正文整体删除，
   正文不出现任何占位符或环境性说明；缺口改记在本文件末尾，供内部决定是否补实验。
3. **H2 的冷启动等价收窄**到 translated 与 identity 两条路径（旧稿的表述覆盖了 native，
   而 online run 并不携带 native cache）。
4. **披露 per-head MLP 同种子不可复现**（同配置同种子两次 run 的 CHG 相差 0.046），
   旧稿"seeded refits reproduce given the seed"的说法在新稿中被替换。
5. **统计口径前置**：表 1 逐行标注 $n$，并在 caption 中区分训练区间与单次区间。

## 尚存的缺口（需要作者决策或更大机器）

以下缺口**不出现在论文正文**，仅作为内部待办（论文已改为只陈述实际所用设置与可支撑的结论）：

| 缺口 | 影响 | 建议 |
|---|---|---|
| 参考口径 500×1,024 校准未跑 | P0-1 只能算"部分完成" | 若有大内存机器，补 3 次 run；否则保持现状表述 |
| text-channel 差异的置信区间缺 artifact | 正文只报点差 $-0.108$，不报区间 | 补一次 bootstrap 并落盘后可加区间 |
| 功效分析缺失 | 正文不再出现"需要更多样本"的说法 | 生成并落盘功效计算后可恢复该句 |
| 4K/8K 行 $n{=}10$、协议不同 | 只能支持"进一步退化"的定性结论 | 论文已如实标注 $n$ 与协议差异 |
| 图片文件名仍含 `oracle` | 仅资产命名，正文已统一为 teacher-content probes | 可在投稿前重命名 |

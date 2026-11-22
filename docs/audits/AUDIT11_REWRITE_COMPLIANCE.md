# AUDIT-11 修订完成度记录

日期：2026-11-22
依据：`docs/audits/audit11.md`（6/10 Weak Accept / Borderline，信心 4/5）
约束：**本轮不做 GPU**——V-permutation 正对照实验与"把功能导向 mapper 放进同一审计"两项
需模型前向/训练，按用户要求排除；其余 P0/P1 全部落地。

## 一、逐项完成情况

| 编号 | 审计意见 | 状态 | 落点 |
|---|---|---|---|
| P0-1 | Appendix E 把近期工作统一解释为"额外重算" | ✅ 重写 | arxiv/iclr/zh 的 `app:concurrent` |
| P0-2 | 三层审计框架应作为第一贡献，解耦降为支持性证据 | ✅ 改定位 | Abstract、§1（贡献句 + "two further results"）、§9 结论、`app:concurrent` |
| P0-3 | Figure 4 图文不一致（右图实为 8 个 octant） | ✅ 改 caption | `fig:probes` caption（arxiv/iclr/zh） |
| P0-4 | Abstract 重复 "help. help." | ✅ 修 | arxiv 摘要（这是我上一轮改摘要时引入的，audit 抓得对） |
| P0-5 | Abstract 的 H1 措辞超出 identity control 能证明的范围 | ✅ 改 | "faithfully preserves a valid student-space cache" |
| P1-1 | §6.1 因果结论超范围 | ✅ 降级 | "insufficient to restore the student's task behavior" + 明确不推断内容与任务无关 |
| P1-2 | Table 4 说 $R^2_V$ "nearly doubles"，实为 +49% | ✅ 改 | `tab:r2` caption：$49\%$（$0.323\to0.481$） |
| P1-3 | Appendix D 暗示 Heo 的结果来自"事后选失败集" | ✅ 改 | 改为客观陈述报告量/单位/校准规模差异 |
| P1-4 | §3.2 非劣样本量论证缺条件 | ✅ 补 | "假定真实效应接近零、方差相当；点估计本身低于 $-\epsilon$ 时扩样本无用" |
| P1-5 | 缺代码/数据可获得性说明 | ✅ 补 | 新增 `REPRODUCIBILITY.md`；论文附录 B 新增 `Availability` 段 |

### P0-1 的具体改法

原文把 CacheLink / KV-Lingo / BridgeKV / RaReCache / ACCT / LatentPort 归为一类，并断言
"它们都是靠花掉严格零 re-prefill 所禁止的计算、或优化接收方而非重建"来恢复能力。审计指出
这条统一解释站不住（KV-Lingo 的蒸馏发生在离线；CacheLink 与 ACCT 仍在零 re-prefill 内）。

新写法分两组陈述，并明确负结果的作用域：

- **仍在严格零 re-prefill 内**：接收方行为导向训练（CacheLink）、蒸馏训练（KV-Lingo）、
  prefill/decode 拆分（BridgeKV）；
- **花掉一小笔目标侧预填**：选择性重算（RaReCache）、固定预算稀疏迁移（ACCT）、
  循环状态搬运（LatentPort）；
- 结论句改为："our negative findings therefore apply to the reconstruction-oriented and
  architecture-informed translators we test, not to all forms of functional cache transport."

§7 里被点名的因果句（"Computation, not more cache, is what recovers the capability"）已删除，
替换为"对照是重建导向 vs 功能导向两类目标，而不是搬运 vs 重算；我们的实验把两者分开，
但并未确立一个可用系统需要哪条路线"。

## 二、未做（及原因）

1. **V-permutation 正对照实验**（审计"可选"项）：需要模型前向 → GPU。本轮排除。
   若后续要做，`REPRODUCIBILITY.md` 第 3 节已列出唯一需要 GPU 的步骤。
2. **把 KV-Lingo / CacheLink 的功能导向 mapper 放进同一 H1/H2/H3 审计**：同属 GPU 范畴。
3. **`paper/humanized/` 未同步**：该副本自 audit-10 起就不再与主稿同源；`REPRODUCIBILITY.md`
   第 7 节已标注"需从 `paper/arxiv/main.tex` 重新改写"。它的 PDF 仍是旧正文。
4. **`docs/paper/CLAIM_MAP.md` 未更新**：新增的 6 条外部引用与 Availability 段尚未登记。

## 三、验证（本机实测）

```
python scripts/paper_consistency_check.py   → PASS 48, FAIL 0
```

五份 PDF 全部重编，0 处未定义引用/交叉引用：

| 文件 | 页数 | 正文位置 |
|---|---|---|
| `paper/arxiv/main.pdf` | 15 | 结论在第 10 页 |
| `paper/iclr2026/main.pdf` | 15 | **结论在第 9 页，附录 A 自第 11 页起 → 9 页限制仍满足** |
| `paper/zh/main.pdf` | 15 | 结论在第 9 页 |
| `paper/humanized/*/main.pdf` | 13 / 13 | 未改动 |

页数取自各稿 `main.aux` 的 `\newlabel{sec:conclusion}`，不依赖 PDF 文本抽取。
本轮新增内容（附录 D/E、Availability、`tab:heo-diff`）全部落在附录，因此 ICLR 正文没有溢出；
总页数从 14 增到 15 是附录变长，不计入 9 页限。

提交包：`arxiv_submission.zip`(109,761 B) / `.tar.gz`(97,643 B) 重打；解压到干净目录实测
`pdflatex + bibtex + pdflatex×2` 退出码 0、15 页、0 未定义引用、0 `not found`，产物与
`paper/arxiv/main.pdf` 逐字节一致。根目录两份分发 PDF 已覆盖（哈希相同）。

## 四、论文的新定位（供后续写作沿用）

> 跨模型 KV 迁移已经有多条技术路线和各自的成功标准；本文提供一套与具体 mapper 无关的
> 审计方法，分开检验注入正确性、接收端 prefill 替代与真正的能力增益。

这条主张现已贯穿 Abstract、§1、§2、§7、Appendix E 与结论；两条解耦在全文统一表述为
"支持性证据"而非"本文发现"。

## 五、不属于本轮修改的事项

- **投稿窗口**：审计指出 ICLR 2027 的投稿截止为 2026-09-25、讨论期修改截至 2026-11-18，
  当前已过。本轮只保证稿件在 9 页正文 + 双盲规则下合规，投递窗口需另定。
- **arXiv 版首页含作者姓名与单位**：这是署名版的设计，不是匿名缺陷；双盲稿在
  `paper/iclr2026/`，作者块默认注释、`\iclrfinalcopy` 关闭。

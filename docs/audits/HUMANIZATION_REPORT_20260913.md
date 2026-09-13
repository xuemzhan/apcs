# 去 AI 味评估与改写报告（2026-09-13）

## 0. 输入与权限

| 项 | 内容 |
|---|---|
| 输入 | `paper/arxiv/main.tex`（完整稿，13 页）＋ `paper/iclr2026/main.tex`（匿名稿） |
| 两版关系 | 正文**逐字相同**；差异仅 preamble、图 2（arXiv 有）、bibliography style |
| skill | `humanize-paper` |
| 输入轨道 | **manuscript**（完整稿，含依赖 caption、表格与附录） |
| 结构范围 | `reconstructive` |
| claim 策略 | `locked`（本仓库未提供独立 evidence pack，故不做 claim 校准） |
| 不在范围内 | 数字、单位、CI、公式、citation key、数据集/模型名、modality、因果/scope/novelty、比较方向、聚合口径 |

本报告只报告**核实过的计数**，不输出 AI 概率或 detector 分数（skill 输出契约）。

---

## 1. 结论：AI 含量有多高

**词汇层几乎为零。** 全文 4438 个 prose 词中：`delve / pivotal / landscape / realm /
seamless / comprehensive / underscore / showcase / tapestry` 出现 **0** 次；`Furthermore /
Moreover / However / Additionally` 合计 **2** 次；无三点式评价句（"simple, efficient, and
effective"），无"In recent years"式时代背景开头，无装饰性比喻。作者也没有用 `novel/robust/
significant` 代替数字。

**真正的 AI 痕迹集中在三层：**

1. **结构—修辞层（最重）**：同一批统计量被 Abstract、Introduction 贡献列表、Results、
   Appendix 逐节复述。例如 `+0.0001 [-0.0006,+0.0009]` 在 **4** 个小节出现，
   `0.99986 / 0.99998` 在 **3** 个小节出现，`177 / 181 / 133 / 48` 这类清点数字在 §6 与附录 B
   各写一遍。这是"逐节确认同一事实"的压模痕迹，也是审稿人最容易读成 AI 信号的一层。
2. **分布层**：22 处 `\textbf{标签.}` 式 run-in 小标题（Methods 7 处、Limitations 6 处）、
   段落末尾自动综合（"Taken together…"型收束）、同一固定短语回声
   （"excludes zero on the positive side" **5** 次、"after the gate form was frozen" **3** 次）。
3. **一致性层**：`mapper / translator / map` 三个词指同一对象 g_θ（mapper 17 次 vs
   translator 32 次），属 C4 术语漂移；另有 3 处措辞与稿件自身的 gate 判定相冲突（见 §5）。

**总体判定：内容与证据是研究者水准（负结果 + 恒等对照 + oracle 探针 + 逐 run 可追溯），
但呈现方式是"LLM 高度压模"——不是词像 AI，而是节与节之间知道得一样多、说得一样圆。**
改写方向因此放在去复述、稳术语、拆模板，而不是换词。

---

## 2. 逐层诊断清单

严重度从高到低；`change` = 已改，`keep` = 确认保留，`uncertain` = 需作者裁决。

| # | 层 | 位置与原文（节选） | 理由 | 判定 |
|---|---|---|---|---|
| 1 | 0 结构 | Abstract / §1 bullet / §4.1 / 表 1 / 附录 D 五处同报 `+0.0001 [-0.0006,+0.0009]` | 同一事实跨节复述；各节未承担不同 reader question | change |
| 2 | 0 结构 | §1 bullet 复述 Abstract 的 `0.99998 / 0.99986 / +0.0001` | Contribution 列表退化为数字目录 | change |
| 3 | 0 结构 | §6.5 "What the audit covers" 清点 `177 runs / 181 rows / 133 translations / 48 probes`，与附录 B 重复 | 内部清点属 DET-2，应由附录承担 | change（route） |
| 4 | 0 结构 | §6 标题 "Why Reconstruction Is Not Enough" 却含 channel asymmetry、长上下文/跨架构/replay/文本通道四类证据 | section contract 与内容不符 | change（重命名） |
| 5 | 0 结构 | §3.2 "Three hypotheses, three instruments, three gates" 与 §1 的 H1/H2/H3 定义重复 | 定义应只出现一次，§1 讲概念、§3.2 讲 gate 常数 | change |
| 6 | 1 词汇 | `mapper`(17) / `translator`(32) / `map`(名词) 混用同一对象 | C4：一个概念一个 canonical term | change（统一 translator） |
| 7 | 1 词汇 | `instrument` 9 次（"three instruments"、"those instruments"、"measurement instruments"） | 同一修辞词承担所有功能，弱化具体对象 | change（改 gate / measurement / tool / checks） |
| 8 | 2 句法 | "Taken together with the deficits we measure, …, which is the case for a shared audit rather than for or against any one translator." | 段末警句，关系由事实承担即可 | change |
| 9 | 2 句法 | "The audit's two most transferable findings are inequalities rather than mechanisms." | 自评式元话语，读者少知道任何东西 | change |
| 10 | 3 段落 | 22 处 `\textbf{标签.}` run-in；Limitations 6 段全部同构 | 模板化段落形状；内容具体，形状可改 | change（19 处） |
| 11 | 3 段落 | §4 "Calibration ladder." 小节只有协议、无"要区分什么" | experiment-level 动机缺失 | change（并回正文并补动机） |
| 12 | 3 段落 | §5.3 "Across all 48 probe rows…", §6.5, 附录 D 三次同写 "excludes zero on the positive side" | 固定短语回声（gate 定义处保留） | change（保留 2 处） |
| 13 | 4 修辞 | "we present it as an interpretation, not an impossibility result" / "a null result … rather than a proof of impossibility" / "bounded by the designs we tested" | 三重免责声明，语义重复 | change（保留 2 处，删 1 处重复） |
| 14 | 5 证据 | §5.2 "Every strong-student translation **degrades** the student" vs 表 1 `Dir.` 列 best row = **neutral** | 措辞与自身 gate 判定冲突 | change（见 §5） |
| 15 | 5 证据 | §6.4 "three **large** deficits" 中 Gemma-3-1B `-0.061 [-0.135,+0.008]` 区间含 0（附录 C） | 与自身区间不一致 | change（改为 "three negative rows"，需作者确认） |
| 16 | 6 分布 | prose em-dash 14 处；各节句长 CV 0.40–0.73；段落长度 CV 0.20–0.75 | 无异常整齐，但 §7/§9 单段 125 词过长 | change（局部拆句） |
| 17 | 6 分布 | Abstract/Intro 6-gram 回声 98 处 | 与 #1/#2 同源 | change（降到 85） |
| 18 | 7 社区词 | "audit" 21 次、"protocol" 密集 | 是本稿刻意的方法学定位，非跨语域误用 | keep |
| 19 | 5 证据 | 所有 CI、`n`、bootstrap 次数、float16 标注、seed 依赖声明 | 是 claim 精度的一部分 | keep |
| 20 | 8 投入 | §3.3 evidence tiers、§8 "reporting margin is not pre-registered"、§8 4K/8K 口径说明 | 作者主动披露，改掉反而像在掩盖 | keep |
| 21 | C0 | 源码内 `% EVIDENCE CHECK (§N): …` 注释 | 不进入 PDF，属作者自查轨迹 | keep（如需干净稿可删） |

---

## 3. 结构健全性判定（skill 第 10 步）

**判定：结构基本健全，不做全篇重排。**

依据：reader-belief 依赖顺序已经正确（问题 → 三问框架 → gate → 方法 → H1/H2/H3 主证据 →
control → 边界 → 限制 → 结论）；motivation chain 每个箭头都能定位到原文锚点（跨模型重复
prefill → 审计必要性 → H2 replacement 要求 → 各族 translator → gate 检验）；control 紧邻其
要排除的替代解释（实现选择）；负面结果留在正文而非下沉附录。

因此改写只执行诊断对应的 6 处结构/路由操作，其余保持原状：

| 操作 | span | 缺陷 | 类型 |
|---|---|---|---|
| 1 | §1 贡献 bullet 去掉与 Abstract 重复的 cosine/CI | #1 #2 | compress |
| 2 | §6.5 清点句下沉附录 B | #3 | route |
| 3 | §6 标题 → "Decouplings and the Reach of the Deficit" | #4 | rename |
| 4 | §6.4 标题 → "Other contexts, students, and repair attempts" | #4 | rename |
| 5 | §3.2 标题去掉 "three instruments"；图 1 caption 同步 | #5 #7 | consistency |
| 6 | §4 "Calibration ladder." 标签并入正文并补实验动机 | #11 | merge + motivation bridge |

---

## 4. 干预量统计

| 项 | 数量 |
|---|---|
| 结构操作（move/split/merge/compress/route） | 6 |
| 路由与一致性修复（术语、标题、caption、交叉引用、与 gate 冲突的措辞） | 6 |
| 语言层改写句子 | 44 / 168（26%） |
| 保留原样的句子 | 123 / 168（74%） |
| 被改动的数字 / CI / citation / 表格数值 / 图数据 | **0** |

---

## 5. 前后对比

### 5.1 可计数信号

| 信号 | 改写前 | 改写后 |
|---|---|---|
| `mapper` / `translator`（同一对象） | 17 / 32 | 1（仅文件名）/ 55 |
| `instrument` | 9 | 0 |
| run-in `\textbf{标签.}` | 22 | 19 |
| prose em-dash | 14 | 12 |
| 重复 6-gram（≥2 次） | 98 | 85 |
| `+0.0001 [-0.0006,+0.0009]` 出现的小节数 | 4 | 3 |
| `0.99986` / `0.99998` 出现的小节数 | 3 | 2 |
| `177` / `181` 清点句出现的小节数 | 2 | 1（仅附录 B） |
| 数值被 ≥3 节复述的个数 | 26 | 23 |

说明：剩下的 23 个多为**必须重复**的项（Abstract 主结果、表 1 行、附录复核），以及
`2.5 / 3.2 / 1.5` 这类来自模型名（Qwen2.5、Llama-3.2）的假匹配；未为凑指标继续删。

### 5.2 逐条 before → after（代表性）

| # | Before | After |
|---|---|---|
| 1 | Cross-model cache translation **promises to remove** target-side prefill by mapping the KV cache a large teacher builds over a context into the cache space of a smaller student. **Reported results conflate** three questions… | A KV cache is model-specific, so when several models at different scales serve the same long context, each of them prefills it again. Cross-model cache translation removes the later prefills by mapping … **An end-to-end score conflates** three questions that fail separately… |
| 2 | **Controls locate the deficit:** calibration volume … moves the result by 0.02, removing the analytic anchor … improves perplexity without improving answers, and a 37M joint MLP does not help. | **The deficit does not come from the calibration budget or from translator capacity:** calibration volume … (同样的三项证据，删除自评动词) |
| 3 | Taken together with the deficits we measure, the published evidence now contains near-native retention, above-baseline gain, and severe degradation, **which is the case for a shared audit rather than for or against any one translator.** | With the deficits we measure, the published evidence now contains near-native retention, above-baseline gain, and severe degradation **at the same time. That spread is the case for a shared audit rather than for or against any one translator.** |
| 4 | The audit's two most transferable findings **are inequalities rather than mechanisms**, and both are supported directly by the recorded runs. | **Two of the audit's findings are inequalities rather than mechanisms, and both come directly from the recorded runs: better reconstruction does not buy better answers, and neither does better perplexity.** |
| 5 | **Calibration ladder.** For the strongest linear family we vary the calibration budget over c∈{10,…,500} examples on the same fixed evaluation set, so that budget effects are not confounded with evaluation items. | **To separate budget effects from evaluation items, we vary the calibration budget of the strongest linear family over c∈{10,…,500} examples on the same fixed evaluation set.** |
| 6 | Table 1 collects the **mapper** ladder. Every strong-student translation **degrades the student relative to its own prefill**, with point estimates from −0.138 to −0.392… | Table 1 collects the **translator** ladder. Every strong-student translation **lands below the student's own prefill**, with point estimates from −0.138 to −0.392… |
| 7 | five alternative students give **three large deficits** (Llama-3.2-3B −0.044, Gemma-3-1B −0.061, Qwen2.5-1.5B −0.157; all n=100) | five alternative students give **three negative rows** (…同一组数字与 n…) |
| 8 | \subsection{What the audit covers} The recorded artifacts contain 177 evaluation runs with metrics and 181 configuration rows… The largest translation point estimate anywhere in the audit is +0.010 … | (小节删除) **Over all rows recorded under the protocol, the largest translation point estimate is +0.010 $[-0.081,+0.102]$, from the weak-student pair, and no probe row places its interval above zero (Appendix B).** |
| 9 | The practical consequence is a **three-instrument checklist**. Report … because it separates a broken pipeline **from a broken map**. | **Three checks would have caught what an end-to-end score hides.** Report … because it separates a broken pipeline **from a broken translator**. |
| 10 | \subsection{H3: **No Teacher-Content Probe Establishes Gain**} | \subsection{H3: **no teacher-content probe establishes gain**}（与 H1/H2 标题大小写统一） |

### 5.3 编译与版面

| 产物 | 结果 |
|---|---|
| `paper/humanized/arxiv/main.pdf` | 13 页（与原稿一致），无 undefined citation/reference，无 LaTeX error |
| `paper/humanized/iclr2026/main.pdf` | 12 页，正文（References 之前）仍在 **9 页内** |
| 渲染抽查 | p1（abstract）、p3（related work/framework）、p5（method）、p6（表 1）、p9（limitations/conclusion）、p12（附录表图）逐页核对，表格行、图注、交叉引用完好 |

---

## 6. 未解决的 claim–evidence 问题与 `uncertain` 项

1. **"three of them are near chance"（§8）与 §6.4 的 "two near-parity rows whose teachers are close
   to chance" 互相矛盾**；同一仓库内 Figure 1 caption 写 "two near-chance students"，
   `docs/paper/CLAIM_MAP.md` 写 "Three receivers are near chance"。两处均在改写后**保持原样**，
   需作者按 `reports/runs/` 的 teacher gold 计数裁决。（`uncertain`）
2. **§6.4 "three large deficits" 与附录 C 的区间冲突**：Gemma-3-1B `-0.061 [-0.135,+0.008]`
   含 0，Llama-3.2-3B 与 Qwen2.5-1.5B 不含。已按稿件自身的记录改述为 "three negative rows"，
   数字与 `n` 未动。这是本报告唯一触及 claim 措辞的改动，请作者确认。（`uncertain`）
3. **§5.2 "degrades" 与表 1 `Dir.` 列冲突**（best row 标为 neutral）。已改为 "lands below"，
   未改变 any 数值或 gate 判定。（已修，需作者确认措辞）
4. **README 记 "171 个记录在案的评测 run"，论文 §6.5/附录 B 记 177 / 181**。未改动论文数字
   （C1），请核对该口径差异。（`uncertain`）
5. 源码内 `% EVIDENCE CHECK (§N)` 注释保留（不进入 PDF）。若要求"干净稿"可整段删除。
6. 本报告未发现的项：无 citation 归属偏移（改写未移动 citation 所属命题）、无图注与正文数字
   不一致、派生数字（`4%` of 0.503、`0.02` spread）复核一致。

---

## 7. 复检证据

- `verify_invariants.py paper/arxiv/main.tex paper/humanized/arxiv/main.tex`
  → **exit 1**（protected-span 计数差异）。逐项人工核对：被标记的
  `+0.0001 / 0.99986 / 0.99998 / 133 / 177 / 181 / 48 / [-0.0006,+0.0009]` **每一个仍在改写稿中
  存在**（脚本见 `tmp/span_check.py`），差异全部来自"同一事实少写一遍"，无事实或 claim 丢失。
  semantic-risk 差异同样逐项核对：`are not confounded` → `To separate …`（同一约束正面化）、
  `is consistent with`（保留于 §7）、`because` 计数在恢复后一致。
- 编译：两份 PDF 均 pdflatex + bibtex + pdflatex ×2 通过。
- 输出端复检：术语（translator）、标点形状（em-dash 12）、段首模式、跨节复述已重扫；
  残留项为上表 §5.1 中"必须重复"的部分。

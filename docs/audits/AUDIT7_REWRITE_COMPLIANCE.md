# audit-7 审稿意见 ↔ 论文逐条比对与处理

**审稿依据**：`docs/audits/audit7_1.md`（第七轮；结论 **Accept 7/10**，reviewer confidence 4/5；
明确说"当前已无足以拒稿的核心方法论漏洞"，并给出 8 条投稿前修改）。
**被检对象**：

- 投稿稿（9 页、匿名）：`paper/cache_audit_rewrite/main.tex`
- arXiv 完整稿（署名、含全部图）：`paper/cache_audit/arxiv/main.tex`

两稿正文同源（arXiv 版由投稿稿生成，另加作者块与图 2/图 4），因此下列修改在两处同时生效。

---

## 一、审稿人要求修改的 8 条

| # | 审稿意见 | 处理 | 位置 | 状态 |
|---|---|---|---|---|
| 1 | H1 的 "exactly / logit cosine = 1 within float tolerance" 不严谨，应给出**明确数值容差**，并最好报告最大 logit 误差 | 改为可复现的数值 gate：**每个被评样本 logit cosine $\ge 0.9998$**；报告实测 **最小值 0.99986、均值 0.99998**；Abstract / §1 / §3.2 / §5.1 / 图 1 全部同步 | 摘要、§1、§3.2、§5.1、图 1 | 完成（容差与实测值均写入；最大 $\ell_\infty$ 误差见下方说明） |
| 2 | `pass/deg/n.e.` 不是互斥标签：non-inferiority 与"显著差于 baseline"可同时成立 | 主结果表把单一 Gate 列拆为 **Repl.**（pass iff CI 下界 $>-\epsilon$）与 **Dir.**（degraded iff CI 上界 $<0$；gain iff 下界 $>0$；否则 neutral）两列；§3.2 增加"两者概念独立、可同时成立"的说明；caption 给出定义 | §3.2、表 1 及 caption | 完成 |
| 3 | Abstract 的 "four further scale pairs" 计数错误 | 改为 "**three further Qwen3 scale pairs (four in total)**" | 摘要（投稿稿与 arXiv 版一致） | 完成 |
| 4 | Method 列了 "low-rank shared-basis maps"，但表格里找不到对应行 | 核实：仓库中 3 个 low-rank run 只有 config、**没有 metrics**（未完成审计协议）→ 从 family 枚举中**删除**该名称（§1 与 §4 同步） | §1、§4 | 完成 |
| 5 | H3 标题 "return nothing" 与统计语言不一致；§7 的 probe-vs-translator 表述过强 | 标题改为 "**H3: No Teacher-Content Probe Establishes Gain**"；§1 小标题同步；§7 改为 "because in our audit the probes reveal no latent benefit that the end-to-end replacement score has hidden." | §2(§7)、§5.3、§1 | 完成 |
| 6 | Conclusion 的 "matching ... composition and calibration length" 易被读成协议完全对齐 | 改为 "matching its source-layer composition and calibration sequence length **under our generated calibration corpus and 100-sequence volume**" | 结论 | 完成 |
| 7 | 补一句普通 CHG 区间的 bootstrap 口径 | 在 §3.1 写明："All intervals use a **paired percentile bootstrap** over evaluation examples with **$10^3$ resamples**; the family-wise probe bound in Section 5.3 uses $10^4$." | §3.1 | 完成（数值以代码为准，见下） |
| 8 | 若干 copy/edit：Figure 3 caption 的 identity/student、Conclusion 的 "not explained by"、MLP 非确定性原因 | ① 图 caption 改为 "The identity control---the student's suffix-native reference---sits in the top-left"；② Conclusion 改为 "the deficit **persists across** the tested calibration budgets, translator capacities, source-layer combinations, and token-alignment controls"；③ 在附录 B 说明：seeds（Python/NumPy/PyTorch、CPU+CUDA）已固定但**未启用 deterministic kernel selection**，残余非确定性来源未定位 | 附录图 4、结论、附录 B | 完成 |

### 与审稿建议不一致的两处（以代码/artifact 为准）

| # | 审稿建议 | 实际情况 | 处理 |
|---|---|---|---|
| 1b | 报告 $\max_i \|z^{(i)}_{\rm inj}-z^{(i)}_{\rm native}\|_\infty$ | 每个 run 的 `capability_score_artifact.json` 只存 **逐样本 `logit_cos_vs_student`**，不存 logits，因此 $\ell_\infty$ 误差无法从现有 artifact 还原 | 改为报告**逐样本 cosine 的最小值**（0.99986）作为等价的最坏情形界；若需要 $\ell_\infty$，要重跑一次并落盘 logits |
| 7b | 建议写 $10^4$ 次重采样 | `apcs/inference/cli.py` 中 gate 用的 `bootstrap_ci(diffs, n_boot=1000)`，即**普通 CHG 区间是 1,000 次**；family-wise max-statistic 才是 10,000 次 | 按实际写入 $10^3$ / $10^4$，避免把建议值当成事实 |

## 二、审稿人明确说"不要再做"的事项（已遵守）

| 事项 | 处理 |
|---|---|
| 不必补 500 条 FineWeb-Edu 才算接受（列为 nice-to-have） | 未扩实验；差异在附录 A 如实列出 |
| 不要为 8K 再去扩 16K/32K | 未扩；4K/8K 仅作"进一步退化"的定性证据 |
| 不要为解决 MLP 非确定性大规模重跑 | 未重跑；保持 range 报告 + 原因说明 |

## 三、版面（审稿人 §17）

| 项 | 结果 |
|---|---|
| 投稿稿正文 | **9 页**（参考文献自第 10 页起），符合 ICLR 正文 ≤9 页；为给新增内容腾位，按审稿建议压缩了 §6.4、text-channel 控制、§7、Limitations |
| 投稿稿匿名 | `\author{Anonymous}`，无作者/机构/邮箱字符串（审稿人看到的实名首页是 arXiv 完整稿，属于有意设置） |
| arXiv 完整稿 | 13 页，图 4 张、表 4 张、参考文献 25 条；含作者块；`main.bbl` 随包，zip 已重建并**从零解压编译验证通过** |

## 四、验证记录

| 检查 | 结果 |
|---|---|
| 投稿稿（`cache_audit_rewrite`）编译 | 通过；9 页正文；无未定义引用；表 1 已为 6 列（新增 Repl./Dir.） |
| arXiv 稿（`cache_audit/arxiv`）编译 | 通过；13 页；无未定义引用；bibtex 无警告（引用 25 条） |
| 提交包自洽性 | 解压 `arxiv_submission.zip` → 编译通过，标题/作者/表格列/H1 容差均在 |
| 术语一致性 | 全文无 "oracle probes"（仅图片文件名保留历史命名）、无 "return nothing"、无 "low-rank shared-basis"、无环境限制类表述 |

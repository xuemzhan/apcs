# audit-8 审稿意见 ↔ 论文逐条比对与处理

**审稿依据**：`docs/audits/audit8_1.md`（第八轮；结论 **Accept 7/10**，reviewer confidence 4/5；
审稿人明确"当前版本已经没有明显的核心方法论漏洞"，给出 8 条投稿前修改清单）。

**被检对象（本轮四个文件同步修改）**：

| 文件 | 形态 | 备注 |
|---|---|---|
| `paper/arxiv/main.tex` | arXiv 完整稿（署名） | 14 页（新增 Appendix B 家族 CI 表）；`arxiv_submission.zip` 已重打包 |
| `paper/iclr2026/main.tex` | ICLR 2026 匿名投稿稿 | 13 页，正文仍在 9 页内（References 自第 9 页起） |
| `paper/humanized/arxiv/main.tex` | 去 AI 味改写副本（署名） | 13 页，正文与上一行同源 |
| `paper/humanized/iclr2026/main.tex` | 去 AI 味改写副本（匿名） | 13 页，正文仍在 9 页内 |

四份正文的**数字、表格、citation 完全一致**；两对之间的差别仍只有 preamble、图 2 与
bibliography style（改写副本另有 §3.2/§5.2/§7/§9 的措辞差异，语义相同）。

---

## 一、审稿人要求的 8 条

| # | 审稿意见 | 处理 | 位置 | 状态 |
|---|---|---|---|---|
| 1（必须） | Conclusion 首句把 primary strong-student 结果推广到 all tested pairs，与 cross-architecture 里 clear margin 的行矛盾 | 首句改为 **"no tested translator establishes replacement for our primary strong-student pair, and the only rows that clear the reporting replacement margin sit on near-chance alternative students and show no measurable gain"**；后文 "the deficit" 限定为 "the deficit on the primary pair" | §9 | 完成 |
| 2（必须） | §6.4 "two near-parity rows whose **teachers** are close to chance" 是事实笔误（teacher 是 Qwen3-4B；接近 chance 的是 receiver） | 改为 **"two near-parity rows on **students** close to chance (Llama-3.2-1B $+0.000$, Gemma-2-2B $-0.009$)"** | §6.4 | 完成 |
| 3（建议） | H1 的 $0.9998$ 阈值未说明是事前固定还是 revision 期间加入，与 $\epsilon$ 的时间线披露不一致 | §3.3 的 confirmatory 列表改为 "the identity gate **in form**（其数值容差 $0.9998$ 是 revision 期间加入的 operationalization）"；Appendix B 增补：**"The identity gate's form was fixed before the runs, while the numerical tolerance 0.9998 was introduced during revision as a reproducible operationalization of the original 'within numerical tolerance' criterion."** 依据：`docs/audits/AUDIT7_REWRITE_COMPLIANCE.md` 记录该容差确为 audit-7 期间加入 | §3.3、附录 B | 完成 |
| 4（建议） | §5.2 "Every strong-student translation degrades" 与表 1 的 `Dir.=neutral` 冲突 | 改为 **"Every strong-student translation has a negative point estimate relative to the student's own prefill, from $-0.138$ to $-0.392$, and all but the best configuration are statistically degraded."** | §5.2 | 完成 |
| 5（建议） | gradient-trained 行只给训练区间却直接写 `degraded`，读者无法审计 $CI_U<0$ | Appendix B 新增表 **"Gradient-trained families: interval detail"**：家族 / 训练次数 / 点估计范围 / **largest CI upper** / 逐训练 CI 上界。实测：per-head MLP c30 5 次训练上界 $-0.148,-0.069,-0.136,-0.231,-0.167$（最大 $-0.069$）；c200 最大 $-0.104$；joint MLP c30 最大 $-0.095$ —— **每个训练的上界都 $<0$**，`degraded` 标签成立 | 附录 B（新表，自动编号 Table 3） | 完成 |
| 6（建议） | 普通 paired bootstrap 从 $10^3$ 提到 $10^4$（靠近 $\epsilon$ 的行会受 Monte-Carlo 噪声影响） | §3.1 改为 "All intervals use a paired percentile bootstrap over evaluation examples with **$10^4$** resamples"；**全部表格与正文区间从 per-sample 记录重算**，见下节 | §3.1 + 全部区间 | 完成 |
| 7（小修） | `training-free probe` 表述不准确（probe 内容来自需拟合的 translator） | Related Work 与 §7 统一改为 **`no-additional-training probe`**，并在 §7 补一句 "the probe takes its content from a fixed, already-calibrated translator and trains nothing new" | §2、§7 | 完成 |
| 8（小修） | Figure 2 未体现本轮新的 Heo concat baseline，易被误读 | 图 2 图注补一句：**"The long-sequence concatenation variants of the closest published design are reported in Table 1 and omitted here for readability."**（ICLR 匿名稿无图 2，无需改） | 图 2（arXiv 两版） | 完成 |

### 与审稿建议不一致 / 未能执行的部分

| # | 审稿建议 | 实际情况 | 处理 |
|---|---|---|---|
| 5（附带） | 报告 $\max_i\|z_{\rm inj}-z_{\rm native}\|_\infty$ 或 mean KL | 录制 artifact 只存逐样本 `logit_cos_vs_student`，**不存 logits**，无法从现有记录还原 | Appendix B 明确写出："the recorded runs store logit cosines rather than logits, so the residual identity difference (minimum $0.99986$) is not decomposed further" |
| 12 | 建议再做一个实验：用 audit protocol 审计一篇公开声称 above-receiver-baseline gain 的方法（如 Universal Context-Reuse Layer） | 需要外部代码复现与新的 GPU 预算，**不是本轮文字修复范围** | 未执行；作为后续实验建议保留 |

---

## 二、$10^3 \to 10^4$ 重采样：方法与影响

**数据来源**：`reports/runs/<run>/inject-eval/inject_eval/capability_score_artifact.json`
的逐样本 `gold_prob`；配对差为 `gold(method) - gold(student)`（同 `sample_id`）。
**计算函数**：仓库自身的 `apcs.metrics.bootstrap_ci`（percentile、seed 0，与
`apcs/inference/cli.py` 调用的同一函数）。
**脚本与 artifact**：`scripts/bootstrap_ci_report.py` → `reports/bootstrap_1e4/ci_1e4.json`
（可用 `--n-boot` 复算，`10^5` 复核稳定性）。

**校验**：

- 重算的**点估计与 `metrics.json` 完全一致**（逐行比对，如 −0.1376、−0.2389、+0.0001），
  说明配对与数据源正确；
- 旧的 $10^3$ 区间与 seed 0 的 $10^3$ 复算相差 0.003–0.02，量级与 $n=30$–$100$ 时
  $10^3$ 次抽样的分位数误差一致——这正是审稿人指出的问题；
- 重算后 45 对区间中 45 对按 3 位小数重印，**全部 38 处正文/表格区间都能被 artifact 解释**
  （仅 identity 行按 4 位小数印出 $[-0.0006,+0.0009]$，与重算值 $[-0.00059,+0.00090]$ 一致）。

**判定稳定性**：所有行的 `Repl.`/`Dir.` 判定**无一处翻转**（按 Table 1 的 `$\epsilon=0.02$`
与上界 $<0$ 规则逐行复核）。跨架构两行在 $10^4$ 下均 clear margin
（Gemma-2-2B rect $-0.0199$、align $-0.0179$；Llama-3.2-1B rect $-0.0116$、align $-0.0120$），
因此附录 C 的 "Alignment changes neither the replacement verdict" 在新数字下**仍然成立**
（旧 $10^3$ 数字下 Gemma-2-2B 未对齐行是 $-0.0202$，反而与该句冲突）。

**移动最大的几处**（点估计不变）：

| 行 | 旧区间 | 新区间 |
|---|---|---|
| Affine per-head, c30, 1.7B | $[-0.317,+0.029]$ | $[-0.312,+0.037]$ |
| Task-aware, c200 | $[-0.336,-0.105]$ | $[-0.336,-0.093]$ |
| Heo concat $k{=}3$ | $[-0.368,-0.084]$ | $[-0.364,-0.081]$ |
| 4B$\to$0.6B affine c200 | $[-0.081,+0.102]$ | $[-0.089,+0.106]$ |
| 1K retrieval | $[-0.534,-0.025]$ | $[-0.549,-0.003]$ |
| 4K retrieval | $[-0.929,-0.170]$ | $[-0.939,-0.135]$ |

---

## 三、遗留边界项（建议作者知悉，未擅自改 claim）

1. **1K retrieval 行的方向判定（degraded）现在很边缘**：上界由 $-0.025$ 变为 $-0.003$，
   按规则 $CI_U<0$ 仍判 degraded，但已接近 0。若审稿人追问，可改述为 "the 1K row is
   degraded by the gate rule, with an upper bound numerically at zero"。
2. **Gemma-2-2B（未对齐）刚好在 margin 上**：下界 $-0.0199$ vs $\epsilon=0.02$，余量 $8\times10^{-5}$。
   论文没有断言"过 margin 的行数"，因此文字不受影响；但不要在 rebuttal 里把它当作稳健结论。
3. **`docs/paper/CLAIM_MAP.md` 中的 CI 是 $10^3$ 版本**，与论文新数字不再逐位一致；如需引用，
   请以 `reports/bootstrap_1e4/ci_1e4.json` 为准（本轮未改动该历史规划文档的数值）。
4. **arXiv 版页数 13 → 14**：新增附录 B 表格与两段说明所致；浮点图 4 独占末页是**改动前就存在**
   的版面形态（`paper/archive` 旧稿同页同样松散），非本轮引入。ICLR 版正文仍为 9 页。
5. **§8 "three of them are near chance" 与 §6.4 "two near-parity rows"**：两处计量单位不同
   （前者数 near-chance receivers，后者数 near-parity 行），审稿人的 item 2 也按此理解；
   本轮只修 typo，未改计数。

---

## 四、复检证据

### 4.1 投稿前 final polish（2026-09-13 冻结轮）

| 项 | 处理 | 位置 |
|---|---|---|
| 删除残留 `n.e.` | §5.2 改为 "it does not pass the replacement gate"；附录 B 改为 "the weak-student pair fails the replacement gate under all three" | §5.2、附录 B |
| "not the lever" 统一改写 | 三处改为 "does not close the gap in the tested range"（calibration volume / capacity and anchoring / source-layer combination） | §5.2 |
| 软化 §6.1 收尾句 | "it is not the part of the cache the student needs to answer" → "does not appear to include the part of the cache the student needs in order to answer" | §6.1 |
| Abstract 范围澄清 | 改为 "The audit protocol records 133 translation configurations across all tested pairs; on the primary strong-student pair the largest point estimate is −0.138, and every interval there lies below zero except the best one, which spans it." | 摘要 |
| §7 中性诊断措辞 | "no latent benefit that the end-to-end replacement score has hidden" → "no probe configuration produces an interval that excludes zero"；同时补上 no-additional-training 的含义（内容取自固定且已校准的 translator，probe 本身不训练任何东西） | §7 |
| 一致性检查发现的算术瑕疵 | 校准阶梯原句引用的两个端点（−0.2511 / −0.2438）只差 0.0073，却接 "a spread of 0.02"；改为 "moves between −0.2569 and −0.2378 across the six budgets, a spread of 0.02"（实测 spread 0.0191） | §5.2 |

自动检查脚本：`scripts/paper_consistency_check.py`（复算 Table 1 的 23 行、probe 表 7 行、
家族 CI 表 3 行、附录全部区间、派生算术、摘要数字、stale 端点、交叉引用与 citation key）。
结果：**四份稿共 48 项全部 PASS，0 FAIL**。

> 勘误：本文件第三节第 7 条的 audit-8 处理说明曾写 §7 已补 "the probe takes its content from
> a fixed, already-calibrated translator and trains nothing new"；复核发现该句在 audit-8 轮
> 实际未落入文件，已在本次 final polish 中补上（现四份稿均含该句）。

### 4.2 编译与版面

- 编译：四个目录均 `pdflatex → bibtex → pdflatex ×2`，**无 undefined citation/reference、
  无 LaTeX Error**。
- 页数：`paper/arxiv` 14、`paper/iclr2026` 13（正文 9 页内）、`paper/humanized/arxiv` 13、
  `paper/humanized/iclr2026` 13（正文 9 页内）。
- 一致性：`scripts/paper_consistency_check.py` 对四个文件逐一复算（Table 1 ×23 行、probe 表 ×7 行、
  家族 CI 表 ×3 行、附录全部区间、派生算术、摘要数字、stale 端点、交叉引用与 citation key），
  **48/48 PASS**；`tmp/verify_ci_after_update.py` 另做一次独立交叉比对，38/38 处区间可由
  `reports/bootstrap_1e4/ci_1e4.json` 解释。
- 残留词检查：四个文件中 `training-free`、`n.e.`、`not the lever`、`latent benefit`、
  `whose teachers are close to chance` 均为 0 次。
- 提交包：`paper/arxiv/arxiv_submission.zip` 已按新 `main.tex` / `main.bbl` 重打包；
  `README_SUBMIT.md` 的 abstract 文本、Comments（14 pages / 4 figures / 5 tables）与变更记录同步更新。

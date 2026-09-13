# arXiv Submission Package — Cache Translation Across Model Scale

## Upload file list（按此清单上传，保持相对路径）

```
main.tex
main.bbl            ← arXiv 不保证运行 bibtex，必须携带
references.bib
figures/fig1_framework.tex       ← 图 1 为 TikZ，被 main.tex 以 \input 引入
figures/fig2_mapper_landscape.pdf
figures/fig3_oracle_probes.pdf
figures/fig4_ppl_acc_decoupling.pdf
```

打包：`tar czf cache_audit_src.tar.gz main.tex main.bbl references.bib figures/*`
（本目录的 `arxiv_submission.zip` 已按上述清单打包，可直接上传。）

## 元数据表单值

| 表单项 | 填写值 |
|---|---|
| Primary subject | cs.CL |
| Cross-list | cs.LG; cs.AI |
| Title | Cache Translation Across Model Scale: An Audit with Identity Controls and Teacher-Content Probes |
| Authors | Xuemin Zhang; Liangbin Hu; Kun Yi; Liheng Zhong; Junpeng Yu |
| Abstract | 见下方纯文本 |
| License | CC BY 4.0（或 arXiv 默认） |
| Comments | 14 pages, 4 figures, 5 tables. Code and run artifacts included in the accompanying repository |
| Journal ref / DOI | 暂无 |

## Abstract（表单纯文本版，无 TeX 命令）

Cross-model cache translation promises to remove target-side prefill by mapping
the key-value (KV) cache a large teacher builds over a context into the cache
space of a smaller student. Reported results conflate three questions that can
fail independently: whether the injection pipeline consumes a foreign cache
correctly (mechanics), whether a mapper replaces the student's own cache at
replacement level (mapping), and whether the teacher's cache carries content the
frozen student can exploit (exploitability). We audit all three under strict
zero re-prefill on Qwen3-4B->1.7B and three further Qwen3 scale pairs (four in
total), using per-sample identity controls, disjoint calibration and evaluation
splits, and gold-letter-probability metrics with bootstrap intervals. Mechanics
passes to a numerical tolerance: every evaluated sample of the identity
injection has logit cosine >= 0.9998 against the student's own prefill (observed
minimum 0.99986, mean 0.99998), with a change of +0.0001 [-0.0006,+0.0009]. No translator we test
replaces the strong student's own prefill. The audit protocol records 133
translation configurations across all tested pairs; on the primary strong-student
pair the largest point estimate is -0.138, and every interval there lies below
zero except the best one, which spans it. Controls locate the deficit: calibration
volume from 10 to 500 examples moves the result by 0.02, removing the analytic
anchor of our most architecture-aware translator improves perplexity without
improving answers, and a 37M-parameter joint MLP does not help. Aligning our
implementation with the closest published design--concatenating the selected
source layers and calibrating on 1,024-token web-text passages--leaves the
verdict unchanged (-0.223 to -0.279). The weak 0.6B student reaches +0.010
[-0.089,+0.106], which establishes neither degradation nor the reporting margin
we adopt (epsilon = 0.02). Teacher-content probes never produce a gain whose
interval excludes zero, and the family-wise 95% bootstrap upper estimate over
the six confirmatory probe configurations is +0.009. Two decouplings survive:
better KV reconstruction (held-out R^2 0.854 for keys, 0.481 for values) does
not improve answers, and near-native perplexity does not either.

## 相对上一版 arXiv 包的变更

- 按 audit-8 意见收口：Conclusion 首句从 "does not replace the student's own prefill"
  收窄为 "no tested translator establishes replacement for our primary strong-student
  pair"，并点明只有 near-chance 替代学生的行能过 reporting margin；§6.4 的
  "whose teachers are close to chance" 事实笔误改为 "on students close to chance"；
  §5.2 改为 "has a negative point estimate … all but the best configuration are
  statistically degraded"；`training-free probe` 统一改为 `no-additional-training probe`；
  Appendix B 补 identity 数值容差的 protocol-timing 说明、`0.9998` 的来源，以及
  gradient-trained 家族的逐训练 CI 上界表（支撑 `degraded` 标签）；图 2 图注注明
  concat Heo 变体只在表 1。
- 普通 paired bootstrap 从 1,000 次重采样统一提升到 10,000 次
  （`scripts/bootstrap_ci_report.py`，artifact `reports/bootstrap_1e4/ci_1e4.json`），
  全部表格与正文区间随之重算；点估计不变，个别端点移动至多 0.02（1,000 次重采样的
  Monte-Carlo 噪声量级）。1K retrieval 行上界由 −0.025 变为 −0.003，方向判定仍为
  degraded。

- 正文按 audit-6 审稿意见重写：标题与全文统一为 *teacher-content probes*；H3 收窄为
  "受测 teacher-derived 内容是否带来增量收益"；补入 CacheBridge 与 Universal
  Context-Reuse Layer 两篇最新工作；family-wise 措辞弱化为 bootstrap 上界；
  主结果表表头改为 `Gate (ε=0.02)`。
- 所有表格数字回到 run 级记录核验；梯度训练族改为跨训练区间。
- 按 audit-7 意见收口：H1 改为可复现的数值容差（每样本 logit cosine ≥ 0.9998，实测最小 0.99986）；
  主结果表把单一 Gate 列拆成 Repl.（替换判定）与 Dir.（方向）两列；摘要的 scale pairs 计数改为
  "three further Qwen3 scale pairs (four in total)"；Method 删除未进入表格的 low-rank family；
  §5.3 标题改为 "No Teacher-Content Probe Establishes Gain"；补一句 CHG 区间为 paired percentile
  bootstrap（1,000 次重采样）。
- 正文不含环境/资源限制类说明，也不含占位符。
- 参考文献 25 条（`main.bbl` 已随包更新）；图 4 张、表 5 张（新增 Appendix B 的
  gradient-trained 家族 CI 上界表）。

- **投稿前 final polish（冻结版）**：① 全文删除残留的 `n.e.` 记号，改成 "does not pass
  the replacement gate" / "fails the replacement gate"；② §5.2 三处 "is not the lever" 改为
  "does not close the gap in the tested range"；③ §6.1 收尾句软化为 "does not appear to
  include the part of the cache the student needs in order to answer"；④ 摘要把 133 个
  translation configuration 的范围写清楚（全部 tested pairs），−0.138 明确限定在 primary
  strong-student pair；⑤ §7 的 "no latent benefit hidden by the end-to-end score" 改为中性的
  "no probe configuration produces an interval that excludes zero"，并补上 probe 为何是
  no-additional-training（内容取自固定且已校准的 translator）；⑥ 校准阶梯那句把 "spread of
  0.02" 的端点改正为实际 min/max（−0.2569 / −0.2378，六个预算，实测 spread 0.0191），原句
  引用的两个端点只差 0.0073，与 0.02 不符。
- **自动一致性检查**：`python3 scripts/paper_consistency_check.py`
  （复算 Table 1 全部 23 行、probe 表 7 行、家族 CI 表 3 行、附录全部区间、派生算术
  4%/2.5 PPL/23 points/0.0191 ladder spread、摘要数字、stale 端点、交叉引用与 citation key）
  → 四份稿 **48/48 项通过，0 失败**。
- 上一版说明里"CHG 区间为 paired percentile bootstrap（1,000 次重采样）"已被本版取代：
  现在统一为 10,000 次重采样。
- 编译：`pdflatex main && bibtex main && pdflatex main && pdflatex main`
  （arXiv 会直接使用随包的 `main.bbl`）。

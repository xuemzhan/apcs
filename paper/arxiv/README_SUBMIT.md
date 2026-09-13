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
| Comments | 13 pages, 4 figures, 4 tables. Code and run artifacts included in the accompanying repository |
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
replaces the strong student's own prefill. Across the 133 translation
configurations recorded under the audit protocol, the largest point estimate on
the strong pair is -0.138 and every strong-student interval lies below zero
except the best one, which spans it. Controls locate the deficit: calibration
volume from 10 to 500 examples moves the result by 0.02, removing the analytic
anchor of our most architecture-aware translator improves perplexity without
improving answers, and a 37M-parameter joint MLP does not help. Aligning our
implementation with the closest published design--concatenating the selected
source layers and calibrating on 1,024-token web-text passages--leaves the
verdict unchanged (-0.223 to -0.279). The weak 0.6B student reaches +0.010
[-0.081,+0.102], which establishes neither degradation nor the reporting margin
we adopt (epsilon = 0.02). Teacher-content probes never produce a gain whose
interval excludes zero, and the family-wise 95% bootstrap upper estimate over
the six confirmatory probe configurations is +0.009. Two decouplings survive:
better KV reconstruction (held-out R^2 0.854 for keys, 0.481 for values) does
not improve answers, and near-native perplexity does not either.

## 相对上一版 arXiv 包的变更

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
- 参考文献 25 条（`main.bbl` 已随包更新）；图 4 张、表 4 张。
- 编译：`pdflatex main && bibtex main && pdflatex main && pdflatex main`
  （arXiv 会直接使用随包的 `main.bbl`）。

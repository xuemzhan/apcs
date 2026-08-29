# arXiv Submission Package — Cache Translation Audit

## Upload file list（按此清单逐个上传，全部平铺相对路径）

```
main.tex
main.bbl            ← arXiv 不保证运行 bibtex，必须携带
references.bib      ← 源文献库（可选但建议）
figures/fig2_mapper_landscape.pdf
figures/fig3_oracle_probes.pdf
figures/fig4_ppl_acc_decoupling.pdf
```

或打包：`tar czf cache_audit_src.tar.gz main.tex main.bbl references.bib figures/*.pdf`

## 元数据表单值

| 表单项 | 填写值 |
|---|---|
| Primary subject | cs.CL |
| Cross-list | cs.LG; cs.AI |
| Title | Cache Translation Across Heterogeneous LLMs: An Audit with Identity Controls and Oracle Bounds |
| Authors | （投稿前替换 main.tex 中的 Anonymous 占位） |
| Abstract | 见下方纯文本 |
| License | CC BY 4.0（或 arXiv 默认） |
| Comments | 11 pages, 3 figures, 2 tables. Code and run artifacts: apcs repository |
| Journal ref / DOI | 暂无 |

## Abstract（表单纯文本版，无 TeX 命令）

Cache translation maps the key-value (KV) cache that a large teacher model
forms over a context into the cache space of a smaller student model, so
that the student answers without re-prefilling the context. A growing line
of work reports quality-preserving translation across heterogeneous models.
We audit this claim on Qwen3 pairs (4B to 1.7B and 4B to 0.6B) with a
three-hypothesis framework and two instruments that prior evaluations omit.
First, an identity control injects the student's own cache through the full
translation-and-injection pipeline: it reproduces the student's own prefill
exactly (per-sample logit cosine 1.000), which isolates mechanics from
mapping. Second, oracle probes mix teacher-cache content into the student's
own cache and bound what any translator could extract. Across five mapper
families, a 200-example calibration ladder, and 21 recorded GPU runs, no
configuration makes the strong student exceed its own prefill:
gold-probability change spans -0.14 to -0.30 against a self-kv control at
+0.000. The weak student reaches parity (+0.010, CI [-0.08, +0.10]). Oracle
probes show a monotone decline as teacher content replaces student content
in early and mid layers, and a perplexity-accuracy decoupling: a translator
that restores near-native fluency (PPL 23.7 versus 21.2) still leaves
accuracy at 0.267 versus 0.500. We conclude that the teacher's
answer-relevant advantage does not survive KV-space translation under zero
re-prefill: it lives in the teacher's weights, not in its cache. We release
the audit protocol as a required checklist for cache-translation claims.

## 已按 arXiv 规范完成的样式项

- `\pdfoutput=1` 显式声明 PDF 输出（arXiv 管线默认 DVI，会导致 TikZ/hyperref 失败）
- `[T1]fontenc + lmodern`：可缩放 Type 1 字体（修复 microtype 与位图 EC 字体的
  致命冲突 "auto expansion is only possible with scalable fonts"）
- `[utf8]inputenc`：显式编码
- 零 Type 3 字体（已验证：24 个字体全部 Type 1 嵌入）
- hyperref 携带 pdftitle/pdfauthor 元数据
- `\input` 已展平（tab1 表格内联进 main.tex）
- 干净环境编译验证：仅上传文件 + pdflatex×2 即可复现（11 页，0 错误，
  引用全解析，bbl 直接生效）
- 宏包全部在 arXiv TeX Live 集合内：geometry/amsmath/graphicx/booktabs/
  multirow/xcolor/tikz/natbib/microtype/hyperref/lmodern

## 投稿前人工确认项

- [ ] 替换 main.tex 作者占位（Anonymous）为真实作者与单位
- [ ] 上传后预览 PDF 首页（标题/作者/摘要）与图 1（TikZ 逻辑图）
- [ ] 若选择 CC 类许可，确认表单 License 一致

# 去 AI 味改写版（humanized）

本目录是 `paper/arxiv/` 与 `paper/iclr2026/` 的**去 AI 味改写副本**（2026-09-13 生成）。
原稿未改动，两版可直接对照 diff。

| 目录 | 对应原稿 | 形态 |
|---|---|---|
| `arxiv/` | `paper/arxiv/` | 13 页完整稿（署名，含图 2 与附录 A–C），`main.pdf` 已编译 |
| `iclr2026/` | `paper/iclr2026/` | 匿名投稿稿，正文仍在 9 页内，`main.pdf` 已编译 |

改写只动**表述、结构与措辞一致性**：数字、置信区间、公式、citation key、数据集/模型名、
modality、因果/scope/novelty、比较方向与聚合口径全部锁定（`humanize-paper` 的
`reconstructive + locked`）。正文动词性差异：两份稿件正文原本逐字相同，改写后仍然逐字相同；
两版的差别仍只有 preamble、图 2 与 bibliography style。

## 编译

```bash
cd paper/humanized/arxiv      && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
cd paper/humanized/iclr2026   && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

两版均无 undefined citation/reference、无 LaTeX error。

## 诊断与逐条改动

评估结论、前后对比、未解决项见 `docs/audits/HUMANIZATION_REPORT_20260913.md`。
如需正式启用，把 `arxiv/main.tex`、`iclr2026/main.tex` 覆盖回 `paper/arxiv/`、`paper/iclr2026/`
并重新生成提交包即可。

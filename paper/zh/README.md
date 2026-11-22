# 中文版（Chinese edition）

`main.tex` 是 `paper/arxiv/main.tex` 的**逐段中文对照翻译**（2026-09-21 生成）。
数字、区间、表格行列、citation key 与 `\label` 与英文原稿一一对应，正文结构
（9 节正文 + 参考文献 + 附录 A–C）保持一致；正文 14 页。

| 项目 | 说明 |
|---|---|
| 翻译底稿 | `paper/arxiv/main.tex`（arXiv 完整稿，14 页） |
| 正文页数 | 14 页（正文 9 页 + 参考文献 + 附录 A–C） |
| 图表 | 图 1 为中文重绘（TikZ，`figures/fig1_framework_zh.tex`）；图 2–4 沿用与英文版**同一批** PDF 图像，图内标注仍为英文 |
| 参考文献 | `references.bib` 与英文版相同，条目保留英文 |
| 数字一致性 | `tmp/zh_number_parity_check.py` 逐项比对中英文稿的数字、citation key 与 `\label`/`\ref` |

## 编译

需要 XeLaTeX + ctex（本机 `wsl` 内 TeX Live 2023，`texlive-xetex` +
`texlive-lang-chinese`）：

```bash
cd paper/zh && xelatex main && bibtex main && xelatex main && xelatex main
```

字体使用 ctex 自带的 Fandol（`fontset=fandol`），不依赖系统字体。

## 与英文版的已知差异

- 图 2–4 的图像文件与英文版相同，因此图内坐标轴与图例文字为英文；图注、正文、
  表格标题均已译为中文。
- 中文版未纳入 `scripts/paper_consistency_check.py` 的自动化核对（该校验按英文
  表格词元解析）；数字一致性由上面的比对脚本与人工对照保证。

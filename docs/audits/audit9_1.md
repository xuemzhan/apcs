# 第九轮独立审查（发布前终审）

日期：2026-09-30
对象：`paper/arxiv/`（arXiv 完整稿）、`paper/iclr2026/`（双盲投稿稿）、
`paper/humanized/`（去 AI 味副本）、`paper/zh/`（中文对照版）、仓库根目录分发件、
`paper/arxiv/arxiv_submission.zip` 提交包。

## 一、总体结论

**可以发布。** 稿件的科学内容、数字、引用、匿名性与页数均通过复核。本轮发现
**1 个发布阻断项**（arXiv 提交包结构损坏，会导致 arXiv 编译失败）与 **1 个分发件
陈旧项**，两者均已在本轮修复并实测验证；另有若干"建议作者确认"的非阻断项。

## 二、已核验且无问题的项

1. **数字与证据一致性。** `python scripts/paper_consistency_check.py` →
   四份英文稿 **48/48 通过，0 失败**：Table 1 全部 23 行、probe 表 7 行、
   家族 CI 表 3 行、附录全部区间、派生算术（ε/基线 ≈ 4%、2.5 PPL 差、23 点准确率
   差、校准阶梯 spread 0.0191）、摘要级数字、stale 端点、交叉引用与 citation key。
2. **聚合口径。** `python scripts/aggregate_claims.py` 输出与正文一致：
   177 个含 metrics 的评测 run、181 个协议行（133 translations + 48 probes）、
   最大翻译点估计 `+0.0101`、清过 ε=0.02 的行 4 行（3 个不同配置）、
   最大 probe 点估计 `+0.0196`、正侧排除零的 probe 行 **0**。
3. **引用真实性。** 17 个 arXiv ID 全部解析且标题匹配；3 处承重引用经原文核验：
   - `heo2026`（2608.03893）：作者列表一致；"closed-form ridge + top-k 源层拼接 +
     de-RoPE + 500×1024 FineWeb-Edu + 73–98% 保留率"与论文表述一致；
   - `cachebridge2026`（2609.00891）：作者一致；"99.83% mean target retention on
     Qwen3 + 一个数量级更少校准数据"一致；
   - `contextreuse2026`（2608.30963）：作者一致；"27.59%→34.48%、高于 1.5B 原生
     基线 6.89 个百分点"一致。
   7 个非 arXiv DOI（SIGCOMM/EuroSys/NeurIPS/EMNLP 等）全部解析且标题匹配。
4. **双盲。** ICLR 稿 `\author{Anonymous}`，作者信息仅存在于注释中；PDF 元数据
   `Author=Anonymous`；正文无身份泄露。
5. **页数。** ICLR 正文 9 页（参考文献自第 10 页起），共 13 页，符合 9 页正文限制；
   arXiv 14 页 / 4 图 / 5 表，与 `README_SUBMIT.md` 的 Comments 字段一致。
6. **中文版。** 与英文稿的 citation key（25/25）、`\label`、`\ref` 完全一致；
   数字差异仅为格式与翻译（如 `$37$M`↔`$3700$万`、`$R^2$` 切分差异），无实质不一致。
7. **分发件同源。** 根目录英文 PDF 与 `paper/arxiv/main.pdf`、根目录中文 PDF 与
   `paper/zh/main.pdf` 哈希一致。

## 三、本轮发现并已修复的问题

### [阻断 · 已修复] arXiv 提交包结构损坏

- **现象：** `paper/arxiv/arxiv_submission.zip` 中 4 个图件被平铺到压缩包根目录，
  而 `main.tex` 以 `\input{figures/fig1_framework}` 与
  `\includegraphics{...{figures/...}}` 引用它们。
- **证据：** git HEAD 版本条目为 `figures/...`，工作区版本为根目录条目 —— 这是
  2026-09-21 署名更新时**重新打包引入的回归**。按 `README_SUBMIT.md` 的清单直接上传
  会编译失败。
- **实测：** 解压后 `pdflatex main.tex` →
  `! LaTeX Error: File 'figures/fig1_framework.tex' not found.`
  `! ==> Fatal error occurred, no output PDF file produced!`
- **处理：** 以与 HEAD 相同的 `figures/` 结构重建压缩包（`main.tex` / `main.bbl` /
  `references.bib` / `README_SUBMIT.md` 与 `paper/arxiv` 逐字节一致），实测
  `pdflatex + bibtex + pdflatex ×2` 退出码 0、产出 14 页、0 处 `not found`；已替换
  仓库中的损坏包。

### [中 · 已修复] humanized 分发 PDF 陈旧

- `paper/humanized/arxiv/Cache Translation ... .pdf` 仍是署名更新前的构建，含占位邮箱
  `huliangbing@xx.com / yikun@xx.com / yujunpeng@xx.com / zhongliheng@xx.com` 与拼错的
  `zmx0813@gmail.com`。
- **处理：** 以 `paper/humanized/arxiv/main.pdf`（13 页、署名正确）覆盖刷新。

## 四、建议作者确认（非阻断）

1. **邮箱与单位不一致：** Liheng Zhong 的邮箱为 `lihengzhong@berkeley.edu`，但
   `\affil` 对全体作者统一写 `CETC`。arXiv 会公开作者列表，建议确认是保留 Berkeley
   邮箱还是改为与单位一致。
2. **ICLR 版本年份：** 目录/README 以 "ICLR 2026" 命名，但当前日期（2026-09-30）已过
   ICLR 2026 投稿周期。若目标是 ICLR 2027，建议确认样式模板与命名
   （`paper/archive/` 中已有 `..._ICLR2027_Style_...` 草稿）。
3. **中文版未纳入自动化检查：** 一致性脚本只解析英文表格词元，中文版由
   `tmp/zh_number_parity_check.py` 与人工对照保证；本轮运行该脚本无实质差异。
4. **citations 形式不统一：** `gist2023`（NeurIPS 2023）与 `stitching2021`
   （NeurIPS 2021）的 `booktitle` 为会议名，但 `doi` 用的是 arXiv DOI
   （`10.48550/...`）。条目真实可解析，仅为形式问题。
5. **计数口径：** 正文 "177 evaluation runs with metrics" 与
   `scripts/aggregate_claims.py` 的定义（`inject-eval/metrics.json` 存在且目录名匹配
   `v12-/v13-/v14-/v15-` 前缀）一致；`README.md` 已注明计数规则。本轮核对无偏差。

## 五、评分（沿用前轮维度）

| 维度 | 评分 |
|---|---|
| Novelty | 8.0 |
| Technical quality | 8.5 |
| Experimental rigor | 9.0 |
| Statistical rigor | 8.5 |
| Related work | 9.0 |
| Claim–evidence alignment | 9.0 |
| Reproducibility | 8.5 |
| Clarity | 8.5 |
| Significance | 8.5 |
| **Overall** | **7 / Accept（发布就绪）** |

科学内容本身无需再改；本轮唯一实质缺陷在**打包与分发环节**，已修复。

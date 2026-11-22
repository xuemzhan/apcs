# 投稿前检查清单

一条命令跑完可自动化的部分：

```bash
python scripts/verify_submission.py        # 纯 CPU，约 3 分钟，退出码 0 才算过
```

它覆盖下面的 A1–A4、B1–B3、C1。其余需要人眼或外部流程确认的，逐项列出。

## A. 稿件

| 项 | 要求 | 现状（2026-11-22） |
|---|---|---|
| A1 | 三份稿编译 0 未定义引用/交叉引用 | ✅ arxiv / iclr2026 / zh 均 0 |
| A2 | ICLR 正文 ≤ 9 页 | ✅ 结论在第 9 页，附录 A 自第 11 页起（总 15 页不计限） |
| A3 | 引用键与 `references.bib` 闭合 | ✅ 三份各 31 键，0 缺失 |
| A4 | 数字与 `reports/runs/` 一致 | ✅ `paper_consistency_check.py` 48/48 |
| A5 | 作者信息与单位、邮箱一致 | ⚠️ 需作者确认：Kun Yi 用 `bit.edu.cn`、Liheng Zhong 用 `berkeley.edu`，而 `\affil` 仍统一写 CETC |
| A6 | 双盲稿无身份泄露 | ✅ `paper/iclr2026`：`\author` 注释、`\iclrfinalcopy` 关闭、PDF 无邮箱串 |
| A7 | 中文版与英文版段落对齐 | ⚠️ 中文版缺 Availability 段与 §8 的 "Several alternative families" 一处措辞，其余已同步 |

## B. 分发件

| 项 | 要求 | 现状 |
|---|---|---|
| B1 | `arxiv_submission.zip` 与源目录逐字节一致 | ✅ 8 个条目，`main.tex`/`main.bbl`/`references.bib` 一致 |
| B2 | 压缩包保留 `figures/` 层级 | ✅ 无平铺图件 |
| B3 | 根目录两份 PDF 与 `paper/arxiv/main.pdf`、`paper/zh/main.pdf` 同哈希 | ✅ |
| B4 | 上传前在干净目录解压试编译 | ✅ 实测 15 页、0 未定义、0 `not found`，产物与主稿字节一致 |

## C. 数字与归档

| 项 | 要求 | 现状 |
|---|---|---|
| C1 | 计数口径可复算 | ✅ `scripts/aggregate_claims.py`：177 runs / 181 行（133 + 48） |
| C2 | 任务指标可由 per-sample 分数重算 | ✅ `paper_consistency_check.py` |
| C3 | 重建 $R^2$ 的来源交代清楚 | ✅ 表 4 注明是 20/10 切分上的**独立诊断重拟合**，不是产生 CHG 的同一 checkpoint；脚本 `scripts/measure_reconstruction_r2.py` |
| C4 | 归档可交给审稿人 | ⚠️ 需在投稿系统上传匿名归档（见 `REPRODUCIBILITY.md` §6） |

## D. 人工确认（脚本无法替代）

1. **投稿窗口**：ICLR 2027 的正式投稿截止为 2026-09-25、讨论期修改截至 2026-11-18；
   当前已过。若投其他会议/期刊，需按目标格式重新核页数与匿名规则。
2. **arXiv 元数据**：`paper/arxiv/README_SUBMIT.md` 的 Comments 已为
   "15 pages, 4 figures, 6 tables"，摘要纯文本已与正文同步；上传前再核一遍
   `Primary subject: cs.CL`、`Cross-list: cs.LG; cs.AI`。
3. **单位署名**：见 A5，需要作者决定是否把两位外部邮箱的作者挂到各自单位。
4. **`humanized/` 是否随投**：该副本自 audit-10 起未同步（`REPRODUCIBILITY.md` §7），
   默认不随投；若要启用，应从 `paper/arxiv/main.tex` 重新改写而不是手工拼接。
5. **不投但仍留在仓库的历史件**：`paper/archive/` 不参与本清单，也不声称与当前数字一致。

## E. 改动后的固定动作

任何改动论文数字、引用或附录之后：

```bash
python scripts/verify_submission.py         # 全绿
# 重新编译三份稿 → 覆盖根目录分发 PDF → 重打 arxiv_submission.zip
python scripts/verify_submission.py         # 再跑一次，确认 B1/B3 不是陈旧件
```

`verify_submission.py` 的 B1/B3 检查会直接指出"打包件落后于源目录"或"分发 PDF 不是最新"，
这两类问题在 audit-9 与 audit-10 各出现过一次。

# 文档索引

本目录存放非代码产出：设计规格、实验协议、逐轮审稿意见、修改方案与投稿清单。

| 路径 | 内容 |
|---|---|
| `SUBMISSION_CHECKLIST.md` | 投稿前逐项清单：页数、匿名、打包、数字、归档 |
| `design/` | `design.md`（设计规格；代码注释里的 "design.md §NN" 指此文件）、`BRIEF.md`、`INNOVATION.md` |
| `protocol/` | `PROTOCOL.md`（实验协议 v1.1→v1.5）、`EXPERIMENT_LOG.md`（历轮实验终审表，含 audit-6 的 4K 勘误） |
| `audits/` | 逐轮审稿意见 `audit1.md`…`audit12.md`；`AUDIT6_GPU_CORRESPONDENCE.md`（六轮意见↔GPU 实验逐条对应）；`AUDIT6/7/8/10/11_REWRITE_COMPLIANCE.md`（各轮意见的逐条落实情况）；`HUMANIZATION_REPORT_20260913.md`（去 AI 味评估与改写报告） |
| `plans/` | 各轮修改方案（`REVISION_PLAN*.md`）、`SUPPLEMENT_EXPERIMENTS_PLAN.md`、`GPU_PLAN_AUDIT6.md`，以及需求/用户故事 |

## 阅读顺序建议

1. `../README.md` —— 项目结论与当前状态
2. `../REPRODUCIBILITY.md` —— 材料在哪、怎么重算、哪些必须 GPU
3. `protocol/PROTOCOL.md` —— 评测协议与 gate 定义
4. `protocol/EXPERIMENT_LOG.md` —— 每轮实验的数字与结论
5. `audits/audit12.md` → `audits/AUDIT12_REWRITE_COMPLIANCE.md` —— 最新一轮意见与落实情况
6. `SUBMISSION_CHECKLIST.md` —— 投稿前照单核对

## 审稿轮次与状态

| 轮次 | 评分 | 状态 |
|---|---|---|
| audit-8 | 7/10 Accept | 定稿 |
| audit-9 | 7/10 Accept | 提交包与分发件修复 |
| audit-10 | 6.5 / minor revision | 定稿后文献复核：修正 §5.2 锚点消融方向、补非劣门功效说明、写死负结果作用域 |
| audit-11 | 7/10 Accept | 重写 Appendix E、把三层框架立为第一贡献、修图注与摘要 |
| audit-12 | 7/10 Accept | 核实 R²↔CHG 的 checkpoint 关系、修正统计范围与复现说明、文字精修 |

## 未完成事项

- 中文版缺 Availability 段，§8 有一处措辞未与英文同步（见 `SUBMISSION_CHECKLIST.md` A7）。
- `paper/humanized/` 自 audit-10 起未同步，启用前需从 `paper/arxiv/main.tex` 重新改写。
- `docs/paper/CLAIM_MAP.md` 未登记 audit-10 之后新增的 6 条外部引用与两个附录小节。
- 作者单位与两位外部邮箱（bit.edu.cn / berkeley.edu）不一致，待作者确认（`SUBMISSION_CHECKLIST.md` A5）。

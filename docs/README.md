# 文档索引

本目录存放非代码产出：设计规格、实验协议、逐轮审稿意见与修改方案。

| 目录 | 内容 |
|---|---|
| `design/` | `design.md`（设计规格；代码注释里的 "design.md §NN" 指此文件）、`BRIEF.md`、`INNOVATION.md` |
| `protocol/` | `PROTOCOL.md`（实验协议 v1.1→v1.5）、`EXPERIMENT_LOG.md`（历轮实验终审表，含 audit-6 的 4K 勘误） |
| `audits/` | 逐轮审稿意见 `audit1.md`…`audit6_1.md`；`AUDIT6_GPU_CORRESPONDENCE.md` 为第六轮意见与 GPU 实验的逐条对应 |
| `plans/` | 各轮修改方案（`REVISION_PLAN*.md`）、`SUPPLEMENT_EXPERIMENTS_PLAN.md`、`GPU_PLAN_AUDIT6.md`，以及需求/用户故事 |

## 阅读顺序建议

1. `../README.md` —— 项目结论与当前状态
2. `protocol/PROTOCOL.md` —— 评测协议与 gate 定义
3. `protocol/EXPERIMENT_LOG.md` —— 每轮实验的数字与结论
4. `audits/audit6_1.md` + `audits/AUDIT6_GPU_CORRESPONDENCE.md` —— 最新一轮审稿意见与其对应的实验
5. `plans/` —— 尚未完成的修改项

## 未完成事项

暂无。论文已按 audit-7 定稿：`paper/arxiv/`（完整稿、署名）与 `paper/iclr2026/`
（正文 9 页、匿名）两版同源，均编译通过、无未定义引用；历史版本在 `paper/archive/`（不入库）。
审稿意见的逐条完成情况见 `audits/AUDIT7_REWRITE_COMPLIANCE.md`。

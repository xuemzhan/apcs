# KVCache-APCS 需求与用户故事

本目录把 `design.md`（2599 行实验方案）拆解为：

1. `requirements.md` — 13 个 Task 的需求分解，按 design.md §71 顺序
2. `user_stories.md` — 13 个用户故事，对应每个 Task 的端到端 CLI 用法
3. `traceability.md` — 用户故事 ↔ design.md 章节 ↔ 代码模块三向追踪矩阵
4. `roadmap.md` — 6 周执行节奏（design.md §68）+ MVP 决策路径（§69）

## 目录

```
plans/
├── README.md           # 本文件
├── requirements.md     # 13 个 Task 的需求（按 §71 顺序）
├── user_stories.md     # 13 个用户故事（端到端 CLI 用法）
├── traceability.md     # 三向追踪矩阵
└── roadmap.md          # 6 周节奏 + MVP 决策
```

## 设计意图

- **`requirements.md`**：从 design.md 各章节抽取"必须做什么"的硬约束，便于任何接手者快速理解"系统该满足哪些条件"。
- **`user_stories.md`**：从最终用户视角写"作为论文作者 / 实验工程师，我需要运行 `python -m apcs.cli tXX` 来做什么"。
- **`traceability.md`**：把用户故事映射到 design.md 章节与代码模块，避免"实现漏掉章节"或"代码无人引用"。
- **`roadmap.md`**：从需求反推 6 周执行计划 + 当下应当选择哪条论文路径（§69 A/B/C/D）。
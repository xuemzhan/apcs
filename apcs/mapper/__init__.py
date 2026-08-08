"""T04/T05/T06 Mapper 子包：Ridge baseline / Replacement / Lightweight low-rank。

包含：
- Ridge Mapper (per-head / de-RoPE / K-V separate) (design.md §20, B3)
- Low-rank base mapper (rank 8/16/32) (design.md §20 Stage 2 / T06)
- Replacement 评估：Retention / KL / Token Agreement (design.md §6, §33)
- PCR 计算 (design.md §3.4)
"""
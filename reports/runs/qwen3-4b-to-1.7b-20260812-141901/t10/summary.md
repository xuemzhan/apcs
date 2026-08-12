# T10 System Cost

> ⚠️ **offline demo**：本任务所有耗时来自线性公式模拟（非 GPU 实测），PSR_A / Cost_B / N_BE 均为推导值，不可作为论文系统收益证据（design.md §38 / §75）。

- VRAM (Teacher est): 0.3 MB
- RAM (Student est): 0.2 MB
- Repeats: 2, Warmup: 1

| ctx | PSR_A p50 | PSR_A p95 | Cost_B p50 (ms) | Cost_B p95 (ms) | N_BE | teacher_KV (MB) | student_KV (MB) |
| --: | --------: | --------: | --------------: | --------------: | ---: | -------------: | --------------: |
| 128 | 0.603 | 0.603 | 11.03 | 11.03 | 4 | 18.0 | 14.0 |

## Single-card pipeline order (§53)
teacher_load → forward → capture → cpu_offload → teacher_unload → cuda_cleanup → student_load → map → inject → decode

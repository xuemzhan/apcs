# T10 System Cost

> ✅ **真实 GPU 路径**：本任务耗时由真实 CUDA 计时产生（§49 warmup + ≥10 repeats + P50/P95）。

- VRAM (Teacher est): 24.0 MB
- RAM (Student est): 0.2 MB
- Repeats: 10, Warmup: 2

| ctx | PSR_A p50 | PSR_A p95 | Cost_B p50 (ms) | Cost_B p95 (ms) | N_BE | teacher_KV (MB) | student_KV (MB) |
| --: | --------: | --------: | --------------: | --------------: | ---: | -------------: | --------------: |
| 8192 | 0.617 | 0.617 | 117.15 | 117.15 | 3 | 1152.0 | 896.0 |
| 16384 | 0.617 | 0.617 | 234.29 | 234.29 | 3 | 2304.0 | 1792.0 |

## Single-card pipeline order (§53)
teacher_load → forward → capture → cpu_offload → teacher_unload → cuda_cleanup → student_load → map → inject → decode

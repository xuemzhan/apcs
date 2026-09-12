# T04 Ridge Baseline

- Calibration samples: 84, context: 512 (auto-sized from available memory; override via cfg.mapper.*)
- Mapper params: 7,340,032
- Mean R²: 0.0034
- Mean KV cosine: 0.7289
- Mean attn-output cosine (§32): 0.8104
- Map latency p50/p95: 1488.13ms / 1506.35ms (§49)
- K/V 独立参数化 (§22): retention_K=0.7289, retention_V=0.7289

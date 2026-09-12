# T04 Ridge Baseline

- Calibration samples: 8, context: 128 (auto-sized from available memory; override via cfg.mapper.*)
- Mapper params: 7,340,032
- Mean R²: 0.2722
- Mean KV cosine: 0.7562
- Mean attn-output cosine (§32): 0.7709
- Map latency p50/p95: 260.13ms / 3739.18ms (§49)
- K/V 独立参数化 (§22): retention_K=0.7561, retention_V=0.7562

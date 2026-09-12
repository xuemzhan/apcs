# T04 Ridge Baseline

- Calibration samples: 20, context: 512 (auto-sized from available memory; override via cfg.mapper.*)
- Mapper params: 7,340,032
- Mean R²: 0.6054
- Mean KV cosine: 0.7497
- Mean attn-output cosine (§32): 0.5685
- Representation diagnostic: REPRESENTATION_PASS (not a task gate)
- Map latency p50/p95: 230.81ms / 3832.18ms (§49)
- K/V 独立参数化 (§22): retention_K=0.9589, retention_V=0.5406

# inject-eval 结果

- **状态**: PASS（四方法评估完成，zero-prefill 验证通过）
- **样本数**: 30（calib/eval 互斥: True）
- **Zero-prefill 验证**: ✅ 通过（真实 cache 审计，协议 v1.1）
- **能力 Gate（gold 口径）**: FAIL（ridge_kv_both: chg=-0.4574841230756786, ci_low=-0.6149580245561659, p=1.0）
- **PSR**: mean=-9.298, median=-8.893（n=30，含测量开销偏保守）

## 各方法指标

| 方法 | accuracy | gold_prob | 置信度 | PPL |
|------|----------|-----------|--------|-----|
| teacher | 0.833 | 0.7503 | 0.8189 | - |
| student | 0.633 | 0.6594 | 0.8284 | 20.40 |
| text | 0.633 | 0.6594 | 0.8284 | - |
| ridge_self_kv | 0.633 | 0.6602 | 0.8284 | 15.61 |
| ridge_native | 0.200 | 0.2335 | 0.7658 | 933033.90 |
| ridge_kv_both | 0.200 | 0.2019 | 0.6696 | 36.60 |

## CHG（gold 口径，ridge_kv_both − student）

- mean=-0.4575, 95% CI=[-0.6150, -0.3024], p=1.0000, TGRR=-5.033
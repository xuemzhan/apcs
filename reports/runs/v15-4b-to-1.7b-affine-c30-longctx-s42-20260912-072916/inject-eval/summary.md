# inject-eval 结果

- **状态**: PASS（四方法评估完成，zero-prefill 验证通过）
- **样本数**: 30（calib/eval 互斥: True）
- **Zero-prefill 验证**: ✅ 通过（真实 cache 审计，协议 v1.1）
- **能力 Gate（gold 口径）**: FAIL（ridge_kv_both: chg=-0.28487202245582866, ci_low=-0.5342845299396691, p=0.977022977022977）
- **PSR**: mean=-187.632, median=-174.350（n=30，含测量开销偏保守）

## 各方法指标

| 方法 | accuracy | gold_prob | 置信度 | PPL |
|------|----------|-----------|--------|-----|
| teacher | 0.733 | 0.6446 | 0.7695 | - |
| student | 0.533 | 0.5168 | 0.8389 | 5.61 |
| text | 0.367 | 0.2889 | 0.5890 | - |
| ridge_self_kv | 0.533 | 0.5169 | 0.8395 | 8.82 |
| ridge_native | 0.267 | 0.2588 | 0.7803 | 10502.38 |
| ridge_kv_both | 0.233 | 0.2319 | 0.9953 | 6554.13 |

## CHG（gold 口径，ridge_kv_both − student）

- mean=-0.2849, 95% CI=[-0.5343, -0.0251], p=0.9770, TGRR=-2.229
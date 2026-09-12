# inject-eval 结果

- **状态**: PASS（四方法评估完成，zero-prefill 验证通过）
- **样本数**: 63（calib/eval 互斥: True）
- **Zero-prefill 验证**: ✅ 通过（真实 cache 审计，协议 v1.1）
- **能力 Gate（gold 口径）**: FAIL（ridge_kv_both: chg=-0.25602378595954434, ci_low=-0.3785828857809638, p=1.0）
- **PSR**: mean=-8.823, median=-8.782（n=63，含测量开销偏保守）

## 各方法指标

| 方法 | accuracy | gold_prob | 置信度 | PPL |
|------|----------|-----------|--------|-----|
| teacher | 0.762 | 0.6752 | 0.8052 | - |
| student | 0.508 | 0.4967 | 0.8286 | 28.18 |
| text | 0.508 | 0.4967 | 0.8286 | - |
| ridge_self_kv | 0.508 | 0.4965 | 0.8285 | 22.74 |
| ridge_native | 0.190 | 0.2124 | 0.7475 | 1441896.25 |
| ridge_kv_both | 0.175 | 0.2407 | 0.6964 | 143.67 |

## CHG（gold 口径，ridge_kv_both − student）

- mean=-0.2560, 95% CI=[-0.3786, -0.1521], p=1.0000, TGRR=-1.435
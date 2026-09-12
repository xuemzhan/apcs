# inject-eval 结果

- **状态**: PASS（四方法评估完成，zero-prefill 验证通过）
- **样本数**: 100（calib/eval 互斥: True）
- **Zero-prefill 验证**: ✅ 通过（真实 cache 审计，协议 v1.1）
- **能力 Gate（gold 口径）**: FAIL（ridge_kv_both: chg=-0.07279451642738467, ci_low=-0.13389792005317572, p=0.998001998001998）
- **PSR**: mean=-5.659, median=-5.591（n=100，含测量开销偏保守）

## 各方法指标

| 方法 | accuracy | gold_prob | 置信度 | PPL |
|------|----------|-----------|--------|-----|
| teacher | 0.770 | 0.6907 | 0.8074 | - |
| student | 0.360 | 0.3180 | 0.7838 | 58.75 |
| text | 0.360 | 0.3180 | 0.7838 | - |
| ridge_self_kv | 0.360 | 0.3178 | 0.7834 | 40.62 |
| ridge_kv_both | 0.250 | 0.2452 | 0.7627 | 49.57 |

## CHG（gold 口径，ridge_kv_both − student）

- mean=-0.0728, 95% CI=[-0.1339, -0.0211], p=0.9980, TGRR=-0.195
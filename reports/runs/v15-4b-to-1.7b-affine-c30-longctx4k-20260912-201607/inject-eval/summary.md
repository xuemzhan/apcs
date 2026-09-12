# inject-eval 结果

- **状态**: PASS（四方法评估完成，zero-prefill 验证通过）
- **样本数**: 20（calib/eval 互斥: True）
- **Zero-prefill 验证**: ✅ 通过（真实 cache 审计，协议 v1.1）
- **能力 Gate（gold 口径）**: FAIL（ridge_kv_both: chg=-0.034598775413726406, ci_low=-0.38017256707426317, p=0.5744255744255744）
- **PSR**: mean=-182.281, median=-158.603（n=20，含测量开销偏保守）

## 各方法指标

| 方法 | accuracy | gold_prob | 置信度 | PPL |
|------|----------|-----------|--------|-----|
| teacher | 0.850 | 0.7256 | 0.7871 | - |
| student | 0.400 | 0.3828 | 0.8075 | 5.57 |
| text | 0.150 | 0.1792 | 0.6003 | - |
| ridge_self_kv | 0.400 | 0.3829 | 0.8075 | 8.67 |
| ridge_native | 0.200 | 0.1592 | 0.7980 | 9513.47 |
| ridge_kv_both | 0.350 | 0.3482 | 0.9962 | 6859.78 |

## CHG（gold 口径，ridge_kv_both − student）

- mean=-0.0346, 95% CI=[-0.3802, +0.3132], p=0.5744, TGRR=-0.101
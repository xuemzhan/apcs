# inject-eval 结果

- **状态**: PASS（四方法评估完成，zero-prefill 验证通过）
- **样本数**: 100（calib/eval 互斥: True）
- **Zero-prefill 验证**: ✅ 通过（真实 cache 审计，协议 v1.1）
- **能力 Gate（gold 口径）**: FAIL（ridge_kv_both: chg=-0.002947262032177108, ci_low=-0.011160447274509376, p=0.7582417582417582）
- **PSR**: mean=-9.787, median=-9.713（n=100，含测量开销偏保守）

## 各方法指标

| 方法 | accuracy | gold_prob | 置信度 | PPL |
|------|----------|-----------|--------|-----|
| teacher | 0.770 | 0.6907 | 0.8074 | - |
| student | 0.300 | 0.2544 | 0.3282 | 30.80 |
| text | 0.300 | 0.2544 | 0.3282 | - |
| ridge_self_kv | 0.290 | 0.2544 | 0.3282 | 23.14 |
| ridge_kv_both | 0.310 | 0.2515 | 0.3123 | 29.05 |

## CHG（gold 口径，ridge_kv_both − student）

- mean=-0.0029, 95% CI=[-0.0112, +0.0053], p=0.7582, TGRR=-0.007
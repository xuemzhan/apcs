# inject-eval 结果

- **状态**: PASS（四方法评估完成，zero-prefill 验证通过）
- **样本数**: 100（calib/eval 互斥: True）
- **Zero-prefill 验证**: ✅ 通过（真实 cache 审计，协议 v1.1）
- **能力 Gate（gold 口径）**: FAIL（ridge_kv_both: chg=-0.006627946888446483, ci_low=-0.018067484449133076, p=0.8731268731268731）
- **PSR**: mean=-6.725, median=-6.234（n=100，含测量开销偏保守）

## 各方法指标

| 方法 | accuracy | gold_prob | 置信度 | PPL |
|------|----------|-----------|--------|-----|
| teacher | 0.770 | 0.6907 | 0.8074 | - |
| student | 0.270 | 0.2573 | 0.3537 | 28.11 |
| text | 0.270 | 0.2573 | 0.3537 | - |
| ridge_self_kv | 0.270 | 0.2574 | 0.3538 | 19.28 |
| ridge_kv_both | 0.190 | 0.2507 | 0.3605 | 24.42 |

## CHG（gold 口径，ridge_kv_both − student）

- mean=-0.0066, 95% CI=[-0.0181, +0.0049], p=0.8731, TGRR=-0.015
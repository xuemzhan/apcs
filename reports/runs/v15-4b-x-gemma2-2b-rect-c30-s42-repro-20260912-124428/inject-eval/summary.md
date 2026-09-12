# inject-eval 结果

- **状态**: PASS（四方法评估完成，zero-prefill 验证通过）
- **样本数**: 100（calib/eval 互斥: True）
- **Zero-prefill 验证**: ✅ 通过（真实 cache 审计，协议 v1.1）
- **能力 Gate（gold 口径）**: FAIL（ridge_kv_both: chg=-0.008873772581704456, ci_low=-0.02016776158686962, p=0.9410589410589411）
- **PSR**: mean=-4.452, median=-4.287（n=100，含测量开销偏保守）

## 各方法指标

| 方法 | accuracy | gold_prob | 置信度 | PPL |
|------|----------|-----------|--------|-----|
| teacher | 0.770 | 0.6907 | 0.8074 | - |
| student | 0.270 | 0.2573 | 0.3537 | 28.11 |
| text | 0.270 | 0.2573 | 0.3537 | - |
| ridge_self_kv | 0.270 | 0.2574 | 0.3538 | 19.28 |
| ridge_kv_both | 0.220 | 0.2484 | 0.3582 | 25.23 |

## CHG（gold 口径，ridge_kv_both − student）

- mean=-0.0089, 95% CI=[-0.0202, +0.0021], p=0.9411, TGRR=-0.020
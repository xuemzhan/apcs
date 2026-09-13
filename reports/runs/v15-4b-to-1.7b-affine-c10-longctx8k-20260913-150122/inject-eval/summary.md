# inject-eval 结果

- **状态**: PASS（四方法评估完成，zero-prefill 验证通过）
- **样本数**: 10（calib/eval 互斥: True）
- **Zero-prefill 验证**: ✅ 通过（真实 cache 审计，协议 v1.1）
- **能力 Gate（gold 口径）**: FAIL（ridge_kv_both: chg=-0.5771260890138461, ci_low=-0.9316603647931202, p=0.985014985014985）
- **PSR**: mean=-218.880, median=-218.698（n=10，含测量开销偏保守）

## 各方法指标

| 方法 | accuracy | gold_prob | 置信度 | PPL |
|------|----------|-----------|--------|-----|
| teacher | 0.900 | 0.7994 | 0.8444 | - |
| student | 0.800 | 0.7801 | 0.8893 | 1.26 |
| text | 0.400 | 0.3393 | 0.6262 | - |
| ridge_self_kv | 0.800 | 0.7796 | 0.8884 | 7.86 |
| ridge_kv_both | 0.200 | 0.2030 | 0.9488 | 24247.43 |

## CHG（gold 口径，ridge_kv_both − student）

- mean=-0.5771, 95% CI=[-0.9317, -0.1184], p=0.9850, TGRR=-29.950
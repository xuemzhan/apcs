# inject-eval 结果

- **状态**: PASS（四方法评估完成，zero-prefill 验证通过）
- **样本数**: 100（calib/eval 互斥: True）
- **Zero-prefill 验证**: ✅ 通过（真实 cache 审计，协议 v1.1）
- **能力 Gate（gold 口径）**: FAIL（ridge_kv_both: chg=-0.15711302631426677, ci_low=-0.2085216259714309, p=1.0）
- **PSR**: mean=-3.918, median=-3.384（n=100，含测量开销偏保守）

## 各方法指标

| 方法 | accuracy | gold_prob | 置信度 | PPL |
|------|----------|-----------|--------|-----|
| teacher | 0.770 | 0.6907 | 0.8074 | - |
| student | 0.400 | 0.3853 | 0.5646 | 20.02 |
| text | 0.400 | 0.3853 | 0.5646 | - |
| ridge_self_kv | 0.410 | 0.3854 | 0.5648 | 15.68 |
| ridge_kv_both | 0.200 | 0.2282 | 0.5784 | 21.78 |

## CHG（gold 口径，ridge_kv_both − student）

- mean=-0.1571, 95% CI=[-0.2085, -0.1059], p=1.0000, TGRR=-0.514
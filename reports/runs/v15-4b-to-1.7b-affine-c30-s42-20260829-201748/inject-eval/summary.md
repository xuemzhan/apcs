# inject-eval 结果

- **状态**: PASS（四方法评估完成，zero-prefill 验证通过）
- **样本数**: 100（calib/eval 互斥: True）
- **Zero-prefill 验证**: ✅ 通过（真实 cache 审计，协议 v1.1）
- **能力 Gate（gold 口径）**: FAIL（ridge_kv_both: chg=-0.24926638717789376, ci_low=-0.3499648331609609, p=1.0）
- **PSR**: mean=-9.006, median=-8.197（n=100，含测量开销偏保守）

## 各方法指标

| 方法 | accuracy | gold_prob | 置信度 | PPL |
|------|----------|-----------|--------|-----|
| teacher | 0.770 | 0.6907 | 0.8074 | - |
| student | 0.520 | 0.5116 | 0.8302 | 29.00 |
| text | 0.520 | 0.5116 | 0.8302 | - |
| ridge_self_kv | 0.520 | 0.5116 | 0.8299 | 22.84 |
| ridge_native | 0.220 | 0.2457 | 0.7523 | 1604489.38 |
| ridge_kv_both | 0.290 | 0.2623 | 0.6830 | 57.63 |

## CHG（gold 口径，ridge_kv_both − student）

- mean=-0.2493, 95% CI=[-0.3500, -0.1487], p=1.0000, TGRR=-1.391
# inject-eval 结果

- **状态**: PASS（四方法评估完成，zero-prefill 验证通过）
- **样本数**: 100（calib/eval 互斥: True）
- **Zero-prefill 验证**: ✅ 通过（真实 cache 审计，协议 v1.1）
- **能力 Gate（gold 口径）**: FAIL（ridge_kv_both: chg=-0.2632360452329668, ci_low=-0.3575519330838328, p=1.0）
- **PSR**: mean=-9.462, median=-8.886（n=100，含测量开销偏保守）

## 各方法指标

| 方法 | accuracy | gold_prob | 置信度 | PPL |
|------|----------|-----------|--------|-----|
| teacher | 0.770 | 0.6907 | 0.8074 | - |
| student | 0.520 | 0.5116 | 0.8302 | 29.00 |
| text | 0.520 | 0.5116 | 0.8302 | - |
| ridge_self_kv | 0.520 | 0.5116 | 0.8299 | 22.84 |
| ridge_native | 0.200 | 0.2424 | 0.7609 | 1668689.62 |
| ridge_kv_both | 0.250 | 0.2483 | 0.6356 | 55.27 |

## CHG（gold 口径，ridge_kv_both − student）

- mean=-0.2632, 95% CI=[-0.3576, -0.1752], p=1.0000, TGRR=-1.469
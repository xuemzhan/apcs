# inject-eval 结果

- **状态**: PASS（四方法评估完成，zero-prefill 验证通过）
- **样本数**: 100（calib/eval 互斥: True）
- **Zero-prefill 验证**: ✅ 通过（真实 cache 审计，协议 v1.1）
- **能力 Gate（gold 口径）**: FAIL（ridge_kv_both: chg=-0.24381759782279303, ci_low=-0.337115555966592, p=1.0）
- **PSR**: mean=-9.983, median=-9.224（n=100，含测量开销偏保守）

## 各方法指标

| 方法 | accuracy | gold_prob | 置信度 | PPL |
|------|----------|-----------|--------|-----|
| teacher | 0.770 | 0.6907 | 0.8074 | - |
| student | 0.520 | 0.5116 | 0.8302 | 29.00 |
| text | 0.520 | 0.5116 | 0.8302 | - |
| ridge_self_kv | 0.520 | 0.5116 | 0.8299 | 22.84 |
| ridge_native | 0.230 | 0.2609 | 0.7674 | 1622753.96 |
| ridge_kv_both | 0.270 | 0.2678 | 0.6667 | 59.20 |

## CHG（gold 口径，ridge_kv_both − student）

- mean=-0.2438, 95% CI=[-0.3371, -0.1424], p=1.0000, TGRR=-1.361
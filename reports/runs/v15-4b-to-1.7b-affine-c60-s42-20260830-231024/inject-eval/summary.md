# inject-eval 结果

- **状态**: PASS（四方法评估完成，zero-prefill 验证通过）
- **样本数**: 100（calib/eval 互斥: True）
- **Zero-prefill 验证**: ✅ 通过（真实 cache 审计，协议 v1.1）
- **能力 Gate（gold 口径）**: FAIL（ridge_kv_both: chg=-0.23779590469158407, ci_low=-0.33843222255645666, p=1.0）
- **PSR**: mean=-9.557, median=-8.148（n=100，含测量开销偏保守）

## 各方法指标

| 方法 | accuracy | gold_prob | 置信度 | PPL |
|------|----------|-----------|--------|-----|
| teacher | 0.770 | 0.6907 | 0.8074 | - |
| student | 0.520 | 0.5116 | 0.8302 | 29.00 |
| text | 0.520 | 0.5116 | 0.8302 | - |
| ridge_self_kv | 0.520 | 0.5116 | 0.8299 | 22.84 |
| ridge_native | 0.230 | 0.2514 | 0.7526 | 1654431.89 |
| ridge_kv_both | 0.260 | 0.2738 | 0.7502 | 62.23 |

## CHG（gold 口径，ridge_kv_both − student）

- mean=-0.2378, 95% CI=[-0.3384, -0.1374], p=1.0000, TGRR=-1.327
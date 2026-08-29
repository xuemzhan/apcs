# inject-eval 结果

- **状态**: PASS（四方法评估完成，zero-prefill 验证通过）
- **样本数**: 20（calib/eval 互斥: True）
- **Zero-prefill 验证**: ✅ 通过（真实 cache 审计，协议 v1.1）
- **能力 Gate（gold 口径）**: FAIL（ridge_kv_both: chg=-0.09613690069153821, ci_low=-0.30683990203013556, p=0.7902097902097902）
- **PSR**: mean=-13.226, median=-12.311（n=20，含测量开销偏保守）

## 各方法指标

| 方法 | accuracy | gold_prob | 置信度 | PPL |
|------|----------|-----------|--------|-----|
| teacher | 0.700 | 0.6284 | 0.8169 | - |
| student | 0.400 | 0.4211 | 0.7220 | 9.54 |
| text | 0.400 | 0.4231 | 0.7261 | - |
| ridge_native | 0.350 | 0.3391 | 0.7991 | 993738.63 |
| ridge_kv_both | 0.250 | 0.3250 | 0.7052 | 147.10 |

## CHG（gold 口径，ridge_kv_both − student）

- mean=-0.0961, 95% CI=[-0.3068, +0.1358], p=0.7902, TGRR=-0.464
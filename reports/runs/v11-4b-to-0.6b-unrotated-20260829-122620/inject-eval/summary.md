# inject-eval 结果

- **状态**: PASS（四方法评估完成，zero-prefill 验证通过）
- **样本数**: 20（calib/eval 互斥: True）
- **Zero-prefill 验证**: ✅ 通过（真实 cache 审计，协议 v1.1）
- **能力 Gate（gold 口径）**: FAIL（ridge_kv_both: chg=0.013124991427951977, ci_low=-0.12043423710439058, p=0.4245754245754246）
- **PSR**: mean=-14.919, median=-14.590（n=20，含测量开销偏保守）

## 各方法指标

| 方法 | accuracy | gold_prob | 置信度 | PPL |
|------|----------|-----------|--------|-----|
| teacher | 0.700 | 0.6284 | 0.8169 | - |
| student | 0.100 | 0.2088 | 0.5587 | 9.90 |
| text | 0.100 | 0.2083 | 0.5560 | - |
| ridge_native | 0.200 | 0.2382 | 0.8094 | 14763.29 |
| ridge_kv_both | 0.200 | 0.2219 | 0.7866 | 223.14 |

## CHG（gold 口径，ridge_kv_both − student）

- mean=+0.0131, 95% CI=[-0.1204, +0.1530], p=0.4246, TGRR=0.031
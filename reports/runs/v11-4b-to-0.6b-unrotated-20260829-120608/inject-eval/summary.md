# inject-eval 结果

- **状态**: FAIL（§52 zero-prefill 验证失败）
- **样本数**: 20（calib/eval 互斥: True）
- **Zero-prefill 验证**: ❌ 失败（真实 cache 审计，协议 v1.1）
- **能力 Gate（gold 口径）**: FAIL（ridge_kv_both: chg=None, ci_low=None, p=None）
- **PSR**: mean=1.000, median=1.000（n=20，含测量开销偏保守）

## 各方法指标

| 方法 | accuracy | gold_prob | 置信度 | PPL |
|------|----------|-----------|--------|-----|
| teacher | 0.700 | 0.6284 | 0.8169 | - |
| student | 0.100 | 0.2088 | 0.5587 | 9.90 |
| text | 0.100 | 0.2083 | 0.5560 | - |
| ridge_native | 0.200 | 0.2382 | 0.8094 | 14763.29 |
| ridge_kv_both | 0.100 | - | 0.0000 | - |
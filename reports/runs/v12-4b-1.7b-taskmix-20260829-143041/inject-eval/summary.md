# inject-eval 结果

- **状态**: FAIL（§52 zero-prefill 验证失败）
- **样本数**: 30（calib/eval 互斥: True）
- **Zero-prefill 验证**: ❌ 失败（真实 cache 审计，协议 v1.1）
- **能力 Gate（gold 口径）**: FAIL（ridge_kv_both: chg=-0.30599412743553356, ci_low=-0.47319357569463477, p=1.0）
- **PSR**: mean=-3.496, median=-3.439（n=30，含测量开销偏保守）

## 各方法指标

| 方法 | accuracy | gold_prob | 置信度 | PPL |
|------|----------|-----------|--------|-----|
| teacher | 0.767 | 0.6934 | 0.8430 | - |
| student | 0.500 | 0.5030 | 0.8572 | 25.36 |
| text | 0.500 | 0.5030 | 0.8572 | - |
| ridge_self_kv | 0.500 | 0.5031 | 0.8568 | 21.23 |
| ridge_native | 0.233 | - | 0.0000 | - |
| ridge_kv_both | 0.233 | 0.1970 | 0.7841 | 18742.55 |
| summary | 0.367 | 0.3717 | 0.7382 | - |

## CHG（gold 口径，ridge_kv_both − student）

- mean=-0.3060, 95% CI=[-0.4732, -0.1291], p=1.0000, TGRR=-1.607
# inject-eval 结果

- **状态**: PASS（四方法评估完成，zero-prefill 验证通过）
- **样本数**: 30（calib/eval 互斥: True）
- **Zero-prefill 验证**: ✅ 通过（真实 cache 审计，协议 v1.1）
- **能力 Gate（gold 口径）**: FAIL（ridge_kv_both: chg=-0.2985863223142544, ci_low=-0.4667736046367416, p=1.0）
- **PSR**: mean=-3.778, median=-3.563（n=30，含测量开销偏保守）

## 各方法指标

| 方法 | accuracy | gold_prob | 置信度 | PPL |
|------|----------|-----------|--------|-----|
| teacher | 0.767 | 0.6934 | 0.8430 | - |
| student | 0.500 | 0.5030 | 0.8572 | 25.36 |
| text | 0.500 | 0.5030 | 0.8572 | - |
| ridge_self_kv | 0.500 | 0.5031 | 0.8568 | 21.23 |
| ridge_native | 0.167 | 0.1688 | 0.6889 | 536082.49 |
| ridge_kv_both | 0.133 | 0.2044 | 0.6790 | 105.98 |
| summary | 0.367 | 0.3717 | 0.7382 | - |

## CHG（gold 口径，ridge_kv_both − student）

- mean=-0.2986, 95% CI=[-0.4668, -0.1331], p=1.0000, TGRR=-1.568
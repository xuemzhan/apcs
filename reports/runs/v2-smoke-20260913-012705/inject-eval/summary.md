# inject-eval 结果

- **状态**: PASS（四方法评估完成，zero-prefill 验证通过）
- **样本数**: 30（calib/eval 互斥: True）
- **Zero-prefill 验证**: ✅ 通过（真实 cache 审计，协议 v1.1）
- **能力 Gate（gold 口径）**: FAIL（ridge_kv_both: chg=-0.2696085015420345, ci_low=-0.5058317088469868, p=0.971028971028971）
- **PSR**: mean=-3.600, median=-2.939（n=30，含测量开销偏保守）

## 各方法指标

| 方法 | accuracy | gold_prob | 置信度 | PPL |
|------|----------|-----------|--------|-----|
| teacher | 0.767 | 0.6934 | 0.8430 | - |
| student | 0.500 | 0.5030 | 0.8572 | 25.36 |
| text | 0.500 | 0.5030 | 0.8572 | - |
| ridge_self_kv | 0.500 | 0.5031 | 0.8568 | 21.23 |
| ridge_kv_both | 0.233 | 0.2334 | 0.9550 | 6966.11 |

## CHG（gold 口径，ridge_kv_both − student）

- mean=-0.2696, 95% CI=[-0.5058, -0.0194], p=0.9710, TGRR=-1.415
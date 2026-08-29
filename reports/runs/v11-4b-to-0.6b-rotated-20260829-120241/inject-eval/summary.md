# inject-eval 结果

- **状态**: PASS（四方法评估完成，zero-prefill 验证通过）
- **样本数**: 20（calib/eval 互斥: True）
- **Zero-prefill 验证**: ✅ 通过（真实 cache 审计，协议 v1.1）
- **能力 Gate（gold 口径）**: FAIL（ridge_kv_both: chg=0.010480931780704288, ci_low=-0.1268343633427243, p=0.4125874125874126）
- **PSR**: mean=-13.636, median=-13.065（n=20，含测量开销偏保守）

## 各方法指标

| 方法 | accuracy | gold_prob | 置信度 | PPL |
|------|----------|-----------|--------|-----|
| teacher | 0.700 | 0.6284 | 0.8169 | - |
| student | 0.100 | 0.2088 | 0.5587 | 9.90 |
| text | 0.100 | 0.2083 | 0.5560 | - |
| ridge_native | 0.200 | 0.2382 | 0.8094 | 14763.29 |
| ridge_kv_both | 0.150 | 0.2193 | 0.7446 | 141.84 |

## CHG（gold 口径，ridge_kv_both − student）

- mean=+0.0105, 95% CI=[-0.1268, +0.1584], p=0.4126, TGRR=0.025
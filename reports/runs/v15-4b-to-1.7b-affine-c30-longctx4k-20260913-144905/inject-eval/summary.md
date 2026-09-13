# inject-eval 结果

- **状态**: PASS（四方法评估完成，zero-prefill 验证通过）
- **样本数**: 10（calib/eval 互斥: True）
- **Zero-prefill 验证**: ✅ 通过（真实 cache 审计，协议 v1.1）
- **能力 Gate（gold 口径）**: FAIL（ridge_kv_both: chg=-0.5963609345390442, ci_low=-0.9289430826467461, p=0.988011988011988）
- **PSR**: mean=-235.156, median=-235.303（n=10，含测量开销偏保守）

## 各方法指标

| 方法 | accuracy | gold_prob | 置信度 | PPL |
|------|----------|-----------|--------|-----|
| teacher | 0.900 | 0.8167 | 0.8786 | - |
| student | 0.900 | 0.7968 | 0.8675 | 1.59 |
| text | 0.400 | 0.3393 | 0.6262 | - |
| ridge_self_kv | 0.900 | 0.7964 | 0.8668 | 8.22 |
| ridge_kv_both | 0.200 | 0.2004 | 0.9922 | 15855.12 |

## CHG（gold 口径，ridge_kv_both − student）

- mean=-0.5964, 95% CI=[-0.9289, -0.1703], p=0.9880, TGRR=-29.886
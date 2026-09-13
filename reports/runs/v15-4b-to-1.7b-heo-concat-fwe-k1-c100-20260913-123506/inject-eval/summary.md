# inject-eval 结果

- **状态**: PASS（四方法评估完成，zero-prefill 验证通过）
- **样本数**: 100（calib/eval 互斥: True）
- **Zero-prefill 验证**: ✅ 通过（真实 cache 审计，协议 v1.1）
- **能力 Gate（gold 口径）**: FAIL（ridge_kv_both: chg=-0.2786944815505555, ci_low=-0.41031748853021266, p=1.0）
- **PSR**: mean=-3.251, median=-2.935（n=100，含测量开销偏保守）

## 各方法指标

| 方法 | accuracy | gold_prob | 置信度 | PPL |
|------|----------|-----------|--------|-----|
| teacher | 0.770 | 0.6907 | 0.8074 | - |
| student | 0.520 | 0.5116 | 0.8302 | 29.00 |
| text | 0.520 | 0.5116 | 0.8302 | - |
| ridge_self_kv | 0.520 | 0.5116 | 0.8299 | 22.84 |
| ridge_kv_both | 0.230 | 0.2329 | 0.9323 | 159446.74 |

## CHG（gold 口径，ridge_kv_both − student）

- mean=-0.2787, 95% CI=[-0.4103, -0.1448], p=1.0000, TGRR=-1.556
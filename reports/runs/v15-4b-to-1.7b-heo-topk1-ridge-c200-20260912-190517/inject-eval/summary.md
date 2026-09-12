# inject-eval 结果

- **状态**: PASS（四方法评估完成，zero-prefill 验证通过）
- **样本数**: 100（calib/eval 互斥: True）
- **Zero-prefill 验证**: ✅ 通过（真实 cache 审计，协议 v1.1）
- **能力 Gate（gold 口径）**: FAIL（ridge_kv_both: chg=-0.27317064349153897, ci_low=-0.37997308152085985, p=1.0）
- **PSR**: mean=-4.296, median=-4.245（n=100，含测量开销偏保守）

## 各方法指标

| 方法 | accuracy | gold_prob | 置信度 | PPL |
|------|----------|-----------|--------|-----|
| teacher | 0.770 | 0.6907 | 0.8074 | - |
| student | 0.520 | 0.5116 | 0.8302 | 29.00 |
| text | 0.520 | 0.5116 | 0.8302 | - |
| ridge_self_kv | 0.520 | 0.5116 | 0.8299 | 22.84 |
| ridge_kv_both | 0.240 | 0.2384 | 0.7880 | 78.49 |

## CHG（gold 口径，ridge_kv_both − student）

- mean=-0.2732, 95% CI=[-0.3800, -0.1577], p=1.0000, TGRR=-1.525
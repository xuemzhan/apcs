# inject-eval 结果

- **状态**: PASS（四方法评估完成，zero-prefill 验证通过）
- **样本数**: 100（calib/eval 互斥: True）
- **Zero-prefill 验证**: ✅ 通过（真实 cache 审计，协议 v1.1）
- **能力 Gate（gold 口径）**: FAIL（ridge_kv_both: chg=-0.06121259776059221, ci_low=-0.13486781099953002, p=0.962037962037962）
- **PSR**: mean=-4.469, median=-4.554（n=100，含测量开销偏保守）

## 各方法指标

| 方法 | accuracy | gold_prob | 置信度 | PPL |
|------|----------|-----------|--------|-----|
| teacher | 0.770 | 0.6907 | 0.8074 | - |
| student | 0.360 | 0.3180 | 0.7838 | 58.75 |
| text | 0.360 | 0.3180 | 0.7838 | - |
| ridge_self_kv | 0.360 | 0.3178 | 0.7834 | 40.62 |
| ridge_kv_both | 0.270 | 0.2568 | 0.8061 | 47.69 |

## CHG（gold 口径，ridge_kv_both − student）

- mean=-0.0612, 95% CI=[-0.1349, +0.0077], p=0.9620, TGRR=-0.164
# inject-eval 结果

- **状态**: PASS（四方法评估完成，zero-prefill 验证通过）
- **样本数**: 100（calib/eval 互斥: True）
- **Zero-prefill 验证**: ✅ 通过（真实 cache 审计，协议 v1.1）
- **能力 Gate（gold 口径）**: FAIL（ridge_kv_both: chg=-0.043695903104068776, ci_low=-0.072913644445734, p=0.999000999000999）
- **PSR**: mean=-5.588, median=-4.912（n=100，含测量开销偏保守）

## 各方法指标

| 方法 | accuracy | gold_prob | 置信度 | PPL |
|------|----------|-----------|--------|-----|
| teacher | 0.770 | 0.6907 | 0.8074 | - |
| student | 0.390 | 0.2810 | 0.3824 | 27.54 |
| text | 0.390 | 0.2810 | 0.3824 | - |
| ridge_self_kv | 0.370 | 0.2810 | 0.3824 | 20.88 |
| ridge_kv_both | 0.200 | 0.2373 | 0.4299 | 33.51 |

## CHG（gold 口径，ridge_kv_both − student）

- mean=-0.0437, 95% CI=[-0.0729, -0.0132], p=0.9990, TGRR=-0.107
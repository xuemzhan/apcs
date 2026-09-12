# inject-eval 结果

- **状态**: PASS（四方法评估完成，zero-prefill 验证通过）
- **样本数**: 100（calib/eval 互斥: True）
- **Zero-prefill 验证**: ✅ 通过（真实 cache 审计，协议 v1.1）
- **能力 Gate（gold 口径）**: FAIL（ridge_kv_both: chg=-0.2044967255585539, ci_low=-0.28401523999568334, p=1.0）
- **PSR**: mean=-11.359, median=-9.984（n=100，含测量开销偏保守）

## 各方法指标

| 方法 | accuracy | gold_prob | 置信度 | PPL |
|------|----------|-----------|--------|-----|
| teacher | 0.720 | 0.6676 | 0.7816 | - |
| student | 0.500 | 0.4948 | 0.7652 | 27.21 |
| text | 0.500 | 0.4948 | 0.7652 | - |
| ridge_self_kv | 0.500 | 0.4944 | 0.7652 | 21.91 |
| ridge_native | 0.220 | 0.2018 | 0.7245 | 2675268.62 |
| ridge_kv_both | 0.260 | 0.2903 | 0.7235 | 58.78 |

## CHG（gold 口径，ridge_kv_both − student）

- mean=-0.2045, 95% CI=[-0.2840, -0.1252], p=1.0000, TGRR=-1.183
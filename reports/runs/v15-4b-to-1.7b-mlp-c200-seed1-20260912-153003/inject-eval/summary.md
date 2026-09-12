# inject-eval 结果

- **状态**: PASS（四方法评估完成，zero-prefill 验证通过）
- **样本数**: 63（calib/eval 互斥: True）
- **Zero-prefill 验证**: ✅ 通过（真实 cache 审计，协议 v1.1）
- **能力 Gate（gold 口径）**: FAIL（ridge_kv_both: chg=-0.22573544197272943, ci_low=-0.3594215451336965, p=0.999000999000999）
- **PSR**: mean=-12.497, median=-10.137（n=63，含测量开销偏保守）

## 各方法指标

| 方法 | accuracy | gold_prob | 置信度 | PPL |
|------|----------|-----------|--------|-----|
| teacher | 0.762 | 0.6752 | 0.8052 | - |
| student | 0.508 | 0.4967 | 0.8286 | 28.18 |
| text | 0.508 | 0.4967 | 0.8286 | - |
| ridge_self_kv | 0.508 | 0.4965 | 0.8285 | 22.74 |
| ridge_native | 0.190 | 0.2124 | 0.7475 | 1441896.25 |
| ridge_kv_both | 0.270 | 0.2710 | 0.7645 | 119.23 |

## CHG（gold 口径，ridge_kv_both − student）

- mean=-0.2257, 95% CI=[-0.3594, -0.1071], p=0.9990, TGRR=-1.265
# inject-eval 结果

- **状态**: PASS（四方法评估完成，zero-prefill 验证通过）
- **样本数**: 63（calib/eval 互斥: True）
- **Zero-prefill 验证**: ✅ 通过（真实 cache 审计，协议 v1.1）
- **能力 Gate（gold 口径）**: FAIL（ridge_kv_both: chg=-0.04147052598140303, ci_low=-0.11997444682960211, p=0.8391608391608392）
- **PSR**: mean=-3.639, median=-3.479（n=63，含测量开销偏保守）

## 各方法指标

| 方法 | accuracy | gold_prob | 置信度 | PPL |
|------|----------|-----------|--------|-----|
| teacher | 0.762 | 0.6752 | 0.8052 | - |
| student | 0.286 | 0.2730 | 0.6706 | 32.01 |
| text | 0.286 | 0.2730 | 0.6706 | - |
| ridge_self_kv | 0.286 | 0.2734 | 0.6706 | 24.39 |
| ridge_native | 0.206 | 0.2396 | 0.7781 | 438149.81 |
| ridge_kv_both | 0.206 | 0.2315 | 0.5501 | 789.97 |

## CHG（gold 口径，ridge_kv_both − student）

- mean=-0.0415, 95% CI=[-0.1200, +0.0379], p=0.8392, TGRR=-0.103
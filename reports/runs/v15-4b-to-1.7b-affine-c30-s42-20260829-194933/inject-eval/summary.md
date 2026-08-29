# inject-eval 结果

- **状态**: PASS（四方法评估完成，zero-prefill 验证通过）
- **样本数**: 232（calib/eval 互斥: True）
- **Zero-prefill 验证**: ✅ 通过（真实 cache 审计，协议 v1.1）
- **能力 Gate（gold 口径）**: FAIL（ridge_kv_both: chg=-0.2445758522940391, ci_low=-0.3086852099235772, p=1.0）
- **PSR**: mean=-9.666, median=-8.284（n=232，含测量开销偏保守）

## 各方法指标

| 方法 | accuracy | gold_prob | 置信度 | PPL |
|------|----------|-----------|--------|-----|
| teacher | 0.759 | 0.6818 | 0.8030 | - |
| student | 0.522 | 0.5045 | 0.8039 | 28.92 |
| text | 0.522 | 0.5052 | 0.8035 | - |
| ridge_self_kv | 0.522 | 0.5044 | 0.8037 | 22.95 |
| ridge_native | 0.263 | 0.2719 | 0.7310 | 3035444.85 |
| ridge_kv_both | 0.284 | 0.2599 | 0.6538 | 55.72 |
| kv_both | - | - | 0.0000 | - |
| native | - | - | 0.0000 | - |
| self_kv | - | - | 0.0000 | - |
| text_handoff | - | - | 0.0000 | - |

## CHG（gold 口径，ridge_kv_both − student）

- mean=-0.2446, 95% CI=[-0.3087, -0.1774], p=1.0000, TGRR=-1.380
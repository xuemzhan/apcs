# inject-eval 结果

- **状态**: PASS（四方法评估完成，zero-prefill 验证通过）
- **样本数**: 100（calib/eval 互斥: True）
- **Zero-prefill 验证**: ✅ 通过（真实 cache 审计，协议 v1.1）
- **能力 Gate（gold 口径）**: FAIL（ridge_kv_both: chg=7.439561363990683e-05, ci_low=-0.011620603738379327, p=0.5054945054945055）
- **PSR**: mean=-7.100, median=-6.209（n=100，含测量开销偏保守）

## 各方法指标

| 方法 | accuracy | gold_prob | 置信度 | PPL |
|------|----------|-----------|--------|-----|
| teacher | 0.770 | 0.6907 | 0.8074 | - |
| student | 0.300 | 0.2544 | 0.3282 | 30.80 |
| text | 0.300 | 0.2544 | 0.3282 | - |
| ridge_self_kv | 0.290 | 0.2544 | 0.3282 | 23.14 |
| ridge_kv_both | 0.260 | 0.2545 | 0.3223 | 30.87 |

## CHG（gold 口径，ridge_kv_both − student）

- mean=+0.0001, 95% CI=[-0.0116, +0.0119], p=0.5055, TGRR=0.000
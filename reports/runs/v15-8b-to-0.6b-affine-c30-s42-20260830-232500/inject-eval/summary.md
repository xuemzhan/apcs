# inject-eval 结果

- **状态**: PASS（四方法评估完成，zero-prefill 验证通过）
- **样本数**: 100（calib/eval 互斥: True）
- **Zero-prefill 验证**: ✅ 通过（真实 cache 审计，协议 v1.1）
- **能力 Gate（gold 口径）**: FAIL（ridge_kv_both: chg=-0.02282787597891986, ci_low=-0.10406927506970004, p=0.7162837162837162）
- **PSR**: mean=-10.023, median=-8.585（n=100，含测量开销偏保守）

## 各方法指标

| 方法 | accuracy | gold_prob | 置信度 | PPL |
|------|----------|-----------|--------|-----|
| teacher | 0.790 | 0.7019 | 0.8100 | - |
| student | 0.280 | 0.2914 | 0.6939 | 32.76 |
| text | 0.280 | 0.2914 | 0.6939 | - |
| ridge_self_kv | 0.280 | 0.2917 | 0.6939 | 24.52 |
| ridge_native | 0.260 | 0.2348 | 0.8103 | 530245.10 |
| ridge_kv_both | 0.270 | 0.2685 | 0.5630 | 99.00 |

## CHG（gold 口径，ridge_kv_both − student）

- mean=-0.0228, 95% CI=[-0.1041, +0.0636], p=0.7163, TGRR=-0.056
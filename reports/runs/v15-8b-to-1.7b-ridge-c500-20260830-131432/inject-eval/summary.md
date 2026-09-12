# inject-eval 结果

- **状态**: PASS（四方法评估完成，zero-prefill 验证通过）
- **样本数**: 100（calib/eval 互斥: True）
- **Zero-prefill 验证**: ✅ 通过（真实 cache 审计，协议 v1.1）
- **能力 Gate（gold 口径）**: FAIL（ridge_kv_both: chg=-0.22406517788448052, ci_low=-0.3204081084597728, p=1.0）
- **PSR**: mean=-6.058, median=-3.359（n=100，含测量开销偏保守）

## 各方法指标

| 方法 | accuracy | gold_prob | 置信度 | PPL |
|------|----------|-----------|--------|-----|
| teacher | 0.790 | 0.7019 | 0.8100 | - |
| student | 0.520 | 0.5116 | 0.8302 | 29.00 |
| text | 0.520 | 0.5116 | 0.8302 | - |
| ridge_self_kv | 0.520 | 0.5116 | 0.8299 | 22.84 |
| ridge_native | 0.260 | 0.2586 | 0.7336 | 2341918.49 |
| ridge_kv_both | 0.310 | 0.2875 | 0.7869 | 146.13 |

## CHG（gold 口径，ridge_kv_both − student）

- mean=-0.2241, 95% CI=[-0.3204, -0.1229], p=1.0000, TGRR=-1.177
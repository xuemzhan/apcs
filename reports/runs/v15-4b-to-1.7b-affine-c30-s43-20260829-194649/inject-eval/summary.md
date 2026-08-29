# inject-eval 结果

- **状态**: PASS（四方法评估完成，zero-prefill 验证通过）
- **样本数**: 29（calib/eval 互斥: True）
- **Zero-prefill 验证**: ✅ 通过（真实 cache 审计，协议 v1.1）
- **能力 Gate（gold 口径）**: FAIL（ridge_kv_both: chg=-0.4661973166501715, ci_low=-0.6266473224170561, p=1.0）
- **PSR**: mean=-10.902, median=-8.739（n=29，含测量开销偏保守）

## 各方法指标

| 方法 | accuracy | gold_prob | 置信度 | PPL |
|------|----------|-----------|--------|-----|
| teacher | 0.793 | 0.7091 | 0.8221 | - |
| student | 0.690 | 0.6678 | 0.8183 | 25.07 |
| text | 0.690 | 0.6686 | 0.8191 | - |
| ridge_self_kv | 0.690 | 0.6675 | 0.8195 | 20.16 |
| ridge_native | 0.138 | 0.1603 | 0.7663 | 2996241.99 |
| ridge_kv_both | 0.207 | 0.2016 | 0.7025 | 49.88 |
| kv_both | - | - | 0.0000 | - |
| native | - | - | 0.0000 | - |
| self_kv | - | - | 0.0000 | - |
| text_handoff | - | - | 0.0000 | - |

## CHG（gold 口径，ridge_kv_both − student）

- mean=-0.4662, 95% CI=[-0.6266, -0.2924], p=1.0000, TGRR=-11.276
# inject-eval 结果

- **状态**: PASS（四方法评估完成，zero-prefill 验证通过）
- **样本数**: 30（calib/eval 互斥: True）
- **Zero-prefill 验证**: ✅ 通过（真实 cache 审计，协议 v1.1）
- **能力 Gate（gold 口径）**: FAIL（ridge_kv_both: chg=-0.3190322060202942, ci_low=-0.47742225915524483, p=0.999000999000999）
- **PSR**: mean=-6.686, median=-6.604（n=30，含测量开销偏保守）

## 各方法指标

| 方法 | accuracy | gold_prob | 置信度 | PPL |
|------|----------|-----------|--------|-----|
| teacher | 0.767 | 0.6934 | 0.8430 | - |
| student | 0.500 | 0.5030 | 0.8572 | 25.36 |
| text | 0.500 | 0.5030 | 0.8572 | - |
| ridge_self_kv | 0.500 | 0.5031 | 0.8568 | 21.23 |
| ridge_native | 0.167 | 0.1688 | 0.6889 | 536082.49 |
| ridge_kv_both | 0.133 | 0.1839 | 0.6885 | 78.58 |

## CHG（gold 口径，ridge_kv_both − student）

- mean=-0.3190, 95% CI=[-0.4774, -0.1621], p=0.9990, TGRR=-1.675
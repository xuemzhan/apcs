# inject-eval 结果

- **状态**: PASS（四方法评估完成，zero-prefill 验证通过）
- **样本数**: 20
- **Zero-prefill 验证**: ✅ 通过

## 各方法均分

| 方法 | 均分 | PPL |
|------|------|-----|
| student | 0.8020 | - |
| teacher | 0.7006 | - |
| text | 0.8020 | - |
| ridge_native | 0.0000 | - |
| ridge_k_only | 0.0000 | - |
| ridge_v_only | 0.0000 | - |
| ridge_kv_both | 0.7223 | 123.49 |
| **CHG** (ridge_kv_both - student) | **-0.0797 | - |
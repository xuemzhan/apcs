# inject-eval 结果

- **状态**: PASS（四方法评估完成，zero-prefill 验证通过）
- **样本数**: 20
- **Zero-prefill 验证**: ✅ 通过

## 各方法均分

| 方法 | 均分 | PPL |
|------|------|-----|
| student | 0.5316 | - |
| teacher | 0.6897 | - |
| text | 0.5316 | - |
| ridge_native | 0.0000 | - |
| ridge_k_only | 0.0000 | - |
| ridge_v_only | 0.0000 | - |
| ridge_kv_both | 0.7123 | 162.51 |
| **CHG** (ridge_kv_both - student) | **+0.1806 | - |
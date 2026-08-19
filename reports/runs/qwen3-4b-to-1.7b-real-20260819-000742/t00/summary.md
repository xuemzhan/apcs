# T00 Compatibility Scanner

- Teacher: Qwen/Qwen3-4B
- Student: Qwen/Qwen3-1.7B
- Verdict: **G1_MATCHED_KV**

Same family + matched KV heads / head_dim. 可直接进入 matched-KV 主实验 (design.md §10).

## Teacher
- layers=36, hidden=2560, heads=32, kv_heads=8, head_dim=128

## Student
- layers=28, hidden=2048, heads=16, kv_heads=8, head_dim=128

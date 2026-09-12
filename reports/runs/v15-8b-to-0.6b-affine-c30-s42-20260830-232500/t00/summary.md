# T00 Compatibility Scanner

- Teacher: /workspace/.cache/modelscope/models/Qwen--Qwen3-8b/snapshots/master (revision=main)
- Student: Qwen/Qwen3-0.6b (revision=main)
- Verdict: **G1_MATCHED_KV**

Same family + matched KV heads / head_dim. 可直接进入 matched-KV 主实验 (design.md §10).

## Teacher
- layers=36, hidden=4096, heads=32, kv_heads=8, head_dim=128, rope_theta=1000000

## Student
- layers=28, hidden=1024, heads=16, kv_heads=8, head_dim=128, rope_theta=1000000

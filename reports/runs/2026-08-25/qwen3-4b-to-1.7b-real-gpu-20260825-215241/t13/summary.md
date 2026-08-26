# T13 Generalization

- Pairs evaluated: 1
- Pairs passing Gate 2A: 0

| run_id | retention | CHG | TGRR | gate2a |
| ------ | --------: | --: | ---: | ------ |
| qwen3-4b-to-1.7b-real-gpu-20260825-215241 | 0.7441 | +0.1626 | +0.5384 | FAIL |

⚠️ 仅 < 2 个 Pair 评估。论文要求 §69 Path A 需在 ≥ 2 个 Large→Small Pair 上同时成立（Table 1）。请运行第二 Pair (`--config configs/pair_smollm2.yaml`)。

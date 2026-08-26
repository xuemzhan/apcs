# Task Report

- TASK_ID: `t13`
- STATUS: **OK**
- Generated at: 2026-08-25T23:10:40.934516

## OBJECTIVE
扫描所有 run_id 的 T05/T09 Gate 2A 状态，输出 Path A 在 ≥ 2 个 Pair 上的可复现性（§11 / §44 / §69）。

## MODEL_PAIR
- Teacher: `Qwen/Qwen3-4B`
- Student: `Qwen/Qwen3-1.7B`

## DATASET
mmlu

## CONFIG
```json
{
  "context_lengths": [
    512,
    1024
  ],
  "seeds": [
    0,
    1,
    2
  ],
  "mapper": {
    "type": "ridge",
    "rank": 32,
    "separate_kv": true,
    "de_rope": true,
    "de_rope_k": true,
    "de_rope_v": false,
    "ridge_lambda_k": 0.001,
    "ridge_lambda_v": 0.01,
    "real_calibration_samples": 32,
    "real_eval_samples": 16,
    "source_top_k": 2,
    "layer_selection": "proportional",
    "alpha_max": 0.5,
    "rms_calibration": true,
    "shared_basis": false,
    "t06_aggregate_samples": 8
  },
  "advantage": {
    "rank": 16,
    "key": "lowrank",
    "value": "lowrank",
    "source_mixer": true,
    "rms_calibration": true,
    "bounded_alpha": true
  }
}
```

## IMPLEMENTATION
T13 Generalization

## OUTPUT_FILES
- `compliance.json`
- `config.json`
- `metadata.json`
- `metrics.json`
- `provider.json`
- `stdout.log`
- `summary.md`

## KEY_METRICS
```json
{
  "task": "T13",
  "n_pairs_evaluated": 1,
  "n_pairs_pass_gate2a": 0,
  "gate2a_rate": 0.0
}
```

## STATISTICAL_CHECK
{
  "n_samples": null,
  "seeds": [
    0,
    1,
    2
  ],
  "ci": 0.95
}

## BEHAVIOR_CHECK
{
  "jcr": "n/a"
}

## GEOMETRY_CHECK
{
  "mean_cka": "n/a"
}

## SYSTEM_COST
{}

## ACCEPTANCE_CRITERIA
{
  "gate": "OK"
}

## FAILURE_ANALYSIS
(无)

## NEXT_ALLOWED_TASK
(全部完成)
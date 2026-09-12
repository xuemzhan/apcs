# Task Report

- TASK_ID: `t07`
- STATUS: **OK**
- Generated at: 2026-08-12T14:38:39.993996

## OBJECTIVE
Teacher-Student gap 分桶（Low/Med/High）；触发 PREREGISTRATION.md 生成（§70）。

## MODEL_PAIR
- Teacher: `Qwen/Qwen3-4B`
- Student: `Qwen/Qwen3-1.7B`

## DATASET
mmlu

## CONFIG
```json
{
  "context_lengths": [
    128
  ],
  "seeds": [
    0,
    1,
    2
  ],
  "mapper": {
    "type": "ridge",
    "rank": 4,
    "separate_kv": true,
    "de_rope": true,
    "source_top_k": 2,
    "layer_selection": "proportional",
    "alpha_max": 0.5,
    "rms_calibration": true,
    "shared_basis": false,
    "t06_aggregate_samples": 2,
    "calibration_samples": 8,
    "replacement_calib_samples": 8,
    "replacement_eval_samples": 4,
    "t04_eval_samples": 4,
    "t06_context": 128,
    "calibration_context": 128,
    "t06_ranks": [
      4,
      8
    ]
  },
  "advantage": {
    "rank": 2,
    "key": "lowrank",
    "value": "lowrank",
    "source_mixer": true,
    "rms_calibration": true,
    "bounded_alpha": true
  }
}
```

## IMPLEMENTATION
T07 Teacher Gap Freeze

## OUTPUT_FILES
- `compliance.json`
- `config.json`
- `metadata.json`
- `metrics.json`
- `stdout.log`
- `summary.md`
- `teacher_gap.json`

## KEY_METRICS
```json
{
  "n_total": 64,
  "n_validation": 32,
  "n_train": 32,
  "gap_distribution": {
    "low": 8,
    "medium": 16,
    "high": 8
  },
  "mean_gap_validation": 0.165388084262023
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
t08
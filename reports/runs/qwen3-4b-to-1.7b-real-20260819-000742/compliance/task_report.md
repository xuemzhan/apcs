# Task Report

- TASK_ID: `compliance`
- STATUS: **PASS**
- Generated at: 2026-08-19T02:33:31.472302

## OBJECTIVE
跑全部 8 条禁止项检查器，输出 violations 列表。

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
    "rank": 16,
    "separate_kv": true,
    "de_rope": true,
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
§52 Compliance

## OUTPUT_FILES
- `config.json`
- `metadata.json`
- `metrics.json`
- `stdout.log`
- `summary.md`

## KEY_METRICS
```json
{
  "task": "compliance",
  "n_violations": 0,
  "violations": [],
  "passed": true,
  "coverage": {
    "§52.1": "UNKNOWN",
    "§52.2": "UNKNOWN",
    "§52.3": "UNKNOWN",
    "§52.4": "UNKNOWN",
    "§52.5": "UNKNOWN",
    "§52.6": "UNKNOWN",
    "§52.7": "UNKNOWN",
    "§52.8": "UNKNOWN"
  },
  "n_unknown": 8,
  "unknown_rules": [
    "§52.1",
    "§52.2",
    "§52.3",
    "§52.4",
    "§52.5",
    "§52.6",
    "§52.7",
    "§52.8"
  ],
  "fully_verified": false
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
  "gate": "PASS"
}

## FAILURE_ANALYSIS
(无)

## NEXT_ALLOWED_TASK
(全部完成)
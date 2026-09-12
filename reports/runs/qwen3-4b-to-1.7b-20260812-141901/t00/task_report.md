# Task Report

- TASK_ID: `t00`
- STATUS: **PASS**
- Generated at: 2026-08-12T14:19:01.045887

## OBJECTIVE
读取 teacher/student 架构，输出 model_compatibility.json 并判定 G1/G2/G3。

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
    "t06_aggregate_samples": 4,
    "calibration_samples": 8,
    "replacement_calib_samples": 8,
    "replacement_eval_samples": 4,
    "t04_eval_samples": 4,
    "t06_context": 128,
    "calibration_context": 128
  },
  "advantage": {
    "rank": 4,
    "key": "lowrank",
    "value": "lowrank",
    "source_mixer": true,
    "rms_calibration": true,
    "bounded_alpha": true
  }
}
```

## IMPLEMENTATION
T00 Compatibility Scanner

## OUTPUT_FILES
- `compliance.json`
- `config.json`
- `metadata.json`
- `metrics.json`
- `model_compatibility.json`
- `stdout.log`
- `summary.md`

## KEY_METRICS
```json
{
  "verdict": "G1_MATCHED_KV",
  "matched_kv": true
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
t01
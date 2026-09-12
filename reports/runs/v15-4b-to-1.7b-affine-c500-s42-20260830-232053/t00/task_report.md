# Task Report

- TASK_ID: `t00`
- STATUS: **PASS**
- Generated at: 2026-08-30T23:21:14.424547

## OBJECTIVE
读取 teacher/student 架构，输出 model_compatibility.json 并判定 G1/G2/G3。

## MODEL_PAIR
- Teacher: `Qwen/Qwen3-4b`
- Student: `Qwen/Qwen3-1.7b`

## DATASET
mmlu

## CONFIG
```json
{
  "context_lengths": [
    512
  ],
  "seeds": [
    42
  ],
  "mapper": {
    "type": "affine",
    "separate_kv": true,
    "de_rope": true,
    "de_rope_k": true,
    "de_rope_v": false,
    "ridge_lambda_k": 0.001,
    "ridge_lambda_v": 0.001,
    "rank": 16,
    "inject_eval_calib_samples": 500,
    "layer_selection": "proportional",
    "inject_eval": true,
    "rope_align": "rotated"
  },
  "advantage": null
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
- `provider.json`
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
    42
  ],
  "ci": null
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
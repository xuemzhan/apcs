# Task Report

- TASK_ID: `inject-eval`
- STATUS: **PASS**
- Generated at: 2026-08-29T01:17:47.528079

## OBJECTIVE
Real path: Teacher prefill → map → inject → Student zero-prefill scoring.

## MODEL_PAIR
- Teacher: `Qwen/Qwen3-4B`
- Student: `Qwen/Qwen3-0.6B`

## DATASET
mmlu

## CONFIG
```json
{
  "context_lengths": [
    512
  ],
  "seeds": [
    456
  ],
  "mapper": {
    "type": "ridge",
    "separate_kv": true,
    "de_rope": true,
    "de_rope_k": true,
    "de_rope_v": false,
    "ridge_lambda_k": 0.001,
    "ridge_lambda_v": 0.001,
    "inject_eval_calib_samples": 20,
    "layer_selection": "proportional",
    "inject_eval": true
  },
  "advantage": null
}
```

## IMPLEMENTATION
Inject Eval

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
  "n_samples": 20,
  "n_capability_records": 80,
  "zero_prefill_verified": true,
  "method_means": {
    "teacher": 0.8105305298863754,
    "student": 0.5886578799500668,
    "text": 0.5886578799500668,
    "ridge_kv_both": 0.7741767867719208
  },
  "ppl_means": {
    "ridge_kv_both": 191.7426019095251
  },
  "chg": 0.185518906821854,
  "ridge_kv_both_score_mean": 0.7741767867719208,
  "student_score_mean": 0.5886578799500668
}
```

## STATISTICAL_CHECK
{
  "n_samples": null,
  "seeds": [
    456
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
(全部完成)
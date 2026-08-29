# Task Report

- TASK_ID: `inject-eval`
- STATUS: **PASS**
- Generated at: 2026-08-29T01:22:53.628385

## OBJECTIVE
Real path: Teacher prefill → map → inject → Student zero-prefill scoring.

## MODEL_PAIR
- Teacher: `Qwen/Qwen3-8B`
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
    123
  ],
  "mapper": {
    "type": "ridge",
    "separate_kv": true,
    "de_rope": true,
    "de_rope_k": true,
    "de_rope_v": false,
    "ridge_lambda_k": 0.0001,
    "ridge_lambda_v": 0.0001,
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
    "teacher": 0.6765788337501016,
    "student": 0.8019910962785147,
    "text": 0.8019910962785147,
    "ridge_kv_both": 0.7803189401120678
  },
  "ppl_means": {
    "ridge_kv_both": 129.85216398808444
  },
  "chg": -0.02167215616644691,
  "ridge_kv_both_score_mean": 0.7803189401120678,
  "student_score_mean": 0.8019910962785147
}
```

## STATISTICAL_CHECK
{
  "n_samples": null,
  "seeds": [
    123
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
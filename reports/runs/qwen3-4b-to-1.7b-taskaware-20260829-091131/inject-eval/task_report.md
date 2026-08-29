# Task Report

- TASK_ID: `inject-eval`
- STATUS: **PASS**
- Generated at: 2026-08-29T09:13:43.498747

## OBJECTIVE
Real path: Teacher prefill → map → inject → Student zero-prefill scoring.

## MODEL_PAIR
- Teacher: `Qwen/Qwen3-4B`
- Student: `Qwen/Qwen3-1.7B`

## DATASET
mmlu

## CONFIG
```json
{
  "context_lengths": [
    512
  ],
  "seeds": [
    0
  ],
  "mapper": {
    "type": "task_aware",
    "separate_kv": true,
    "de_rope": true,
    "de_rope_k": true,
    "de_rope_v": false,
    "ridge_lambda_k": 0.001,
    "ridge_lambda_v": 0.001,
    "lr": "5e-4",
    "n_finetune_steps": 30,
    "inject_eval_calib_samples": 50,
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
    "teacher": 0.7809207201554706,
    "student": 0.6907364408198028,
    "text": 0.6907364408198028,
    "ridge_kv_both": 0.7941371411341396
  },
  "ppl_means": {
    "ridge_kv_both": 131.41170051023673
  },
  "chg": 0.10340070031433679,
  "ridge_kv_both_score_mean": 0.7941371411341396,
  "student_score_mean": 0.6907364408198028
}
```

## STATISTICAL_CHECK
{
  "n_samples": null,
  "seeds": [
    0
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
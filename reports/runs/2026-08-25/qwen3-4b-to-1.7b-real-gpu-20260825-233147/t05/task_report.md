# Task Report

- TASK_ID: `t05`
- STATUS: **FAIL**
- Generated at: 2026-08-25T23:35:20.208698

## OBJECTIVE
跨模型 KV 替换 + Retention + KL + Token Agreement + Latency（Gate 1）。

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
    "alpha_max": 0.5,
    "de_rope": true,
    "de_rope_k": true,
    "de_rope_v": false,
    "layer_selection": "proportional",
    "rank": 64,
    "real_calibration_samples": 64,
    "real_eval_samples": 16,
    "ridge_lambda_k": 0.001,
    "ridge_lambda_v": 0.001,
    "rms_calibration": true,
    "separate_kv": true,
    "shared_basis": false,
    "source_top_k": 4,
    "t06_aggregate_samples": 8,
    "type": "ridge"
  },
  "advantage": {
    "bounded_alpha": true,
    "key": "lowrank",
    "rank": 16,
    "rms_calibration": true,
    "source_mixer": true,
    "value": "lowrank"
  }
}
```

## IMPLEMENTATION
T05 Replacement

## OUTPUT_FILES
- `compliance.json`
- `config.json`
- `kv_split_manifest.json`
- `metadata.json`
- `metrics.json`
- `provider.json`
- `stdout.log`
- `summary.md`

## KEY_METRICS
```json
{
  "task": "T05",
  "contexts": [
    70
  ],
  "retention_per_context": [
    {
      "context": 70,
      "retention": 0.7716372947759222,
      "token_agreement": 0.53125,
      "retention_K": 0.9660981994607238,
      "retention_V": 0.5771763900911207,
      "token_agreement_K": 0.8046875,
      "token_agreement_V": 0.2578125
    }
  ],
  "mean_retention": 0.7716372947759222,
  "mean_token_agreement": 0.53125,
  "latency_p50_ms": 6474.904261063784,
  "latency_p95_ms": 7678.39567088522,
  "repeats": 10,
  "warmup": 2,
  "gap_strata": {
    "low": [],
    "medium": [
      0.7716372947759222
    ],
    "high": []
  },
  "gate1": "FAIL",
  "data_source": "hf",
  "offline_demo": false,
  "de_rope_k": true,
  "de_rope_v": false,
  "ridge_lambda_k": 0.001,
  "ridge_lambda_v": 0.001,
  "mean_retention_K": 0.9660981994607238,
  "mean_retention_V": 0.5771763900911207,
  "mean_token_agreement_K": 0.8046875,
  "mean_token_agreement_V": 0.2578125
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
  "gate": "FAIL"
}

## FAILURE_ANALYSIS
(无)

## NEXT_ALLOWED_TASK
(全部完成)
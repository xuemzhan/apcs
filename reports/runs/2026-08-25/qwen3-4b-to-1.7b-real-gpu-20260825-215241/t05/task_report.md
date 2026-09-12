# Task Report

- TASK_ID: `t05`
- STATUS: **FAIL**
- Generated at: 2026-08-25T22:11:03.208686

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
    "type": "ridge",
    "rank": 16,
    "separate_kv": true,
    "de_rope": true,
    "de_rope_k": true,
    "de_rope_v": false,
    "ridge_lambda_k": 0.001,
    "ridge_lambda_v": 0.001,
    "real_calibration_samples": 16,
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
    78
  ],
  "retention_per_context": [
    {
      "context": 78,
      "retention": 0.744114871498049,
      "token_agreement": 0.578125,
      "retention_K": 0.9625840000286432,
      "retention_V": 0.5256457429674548,
      "token_agreement_K": 0.8359375,
      "token_agreement_V": 0.3203125
    }
  ],
  "mean_retention": 0.744114871498049,
  "mean_token_agreement": 0.578125,
  "latency_p50_ms": 2200.7125155068934,
  "latency_p95_ms": 2581.9752521347255,
  "repeats": 10,
  "warmup": 2,
  "gap_strata": {
    "low": [],
    "medium": [
      0.744114871498049
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
  "mean_retention_K": 0.9625840000286432,
  "mean_retention_V": 0.5256457429674548,
  "mean_token_agreement_K": 0.8359375,
  "mean_token_agreement_V": 0.3203125
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
# Task Report

- TASK_ID: `t05`
- STATUS: **FAIL**
- Generated at: 2026-08-12T14:20:50.613057

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
T05 Replacement

## OUTPUT_FILES
- `compliance.json`
- `config.json`
- `metadata.json`
- `metrics.json`
- `stdout.log`
- `summary.md`

## KEY_METRICS
```json
{
  "task": "T05",
  "contexts": [
    128
  ],
  "retention_per_context": [
    {
      "context": 128,
      "retention": 0.7561930237213699,
      "token_agreement": 0.328125,
      "retention_K": 0.7561440673709449,
      "retention_V": 0.7562419800717949,
      "token_agreement_K": 0.3125,
      "token_agreement_V": 0.34375
    }
  ],
  "mean_retention": 0.7561930237213699,
  "mean_token_agreement": 0.328125,
  "latency_p50_ms": 4009.2905499914195,
  "latency_p95_ms": 4383.492144997581,
  "repeats": 2,
  "warmup": 1,
  "gap_strata": {
    "low": [],
    "medium": [
      0.7561930237213699
    ],
    "high": []
  },
  "gate1": "FAIL",
  "mean_retention_K": 0.7561440673709449,
  "mean_retention_V": 0.7562419800717949,
  "mean_token_agreement_K": 0.3125,
  "mean_token_agreement_V": 0.34375
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
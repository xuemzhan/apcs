# Task Report

- TASK_ID: `t05`
- STATUS: **FAIL**
- Generated at: 2026-08-19T01:31:56.560389

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
- `metadata.json`
- `metrics.json`
- `stdout.log`
- `summary.md`

## KEY_METRICS
```json
{
  "task": "T05",
  "contexts": [
    512,
    1024
  ],
  "retention_per_context": [
    {
      "context": 512,
      "retention": 0.7280305691417613,
      "token_agreement": 0.21250000000000002,
      "retention_K": 0.7280303236810618,
      "retention_V": 0.7280308146024609,
      "token_agreement_K": 0.23125,
      "token_agreement_V": 0.19375
    },
    {
      "context": 1024,
      "retention": 0.6997685760818326,
      "token_agreement": 0.20625,
      "retention_K": 0.6997616148957578,
      "retention_V": 0.6997755372679075,
      "token_agreement_K": 0.2,
      "token_agreement_V": 0.2125
    }
  ],
  "mean_retention": 0.7138995726117969,
  "mean_token_agreement": 0.209375,
  "latency_p50_ms": 148431.9669980323,
  "latency_p95_ms": 189057.70263357554,
  "repeats": 10,
  "warmup": 2,
  "gap_strata": {
    "low": [
      0.6997685760818326
    ],
    "medium": [],
    "high": [
      0.7280305691417613
    ]
  },
  "gate1": "FAIL",
  "mean_retention_K": 0.7138959692884098,
  "mean_retention_V": 0.7139031759351842,
  "mean_token_agreement_K": 0.215625,
  "mean_token_agreement_V": 0.203125
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
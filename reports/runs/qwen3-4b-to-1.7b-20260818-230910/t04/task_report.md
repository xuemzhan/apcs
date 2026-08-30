# Task Report

- TASK_ID: `t04`
- STATUS: **PASS**
- Generated at: 2026-08-18T23:27:44.496507

## OBJECTIVE
Full Ridge 校准，输出 R² / KV cosine / attn-output cosine。

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
    "t06_aggregate_samples": 2,
    "calibration_samples": 8,
    "replacement_calib_samples": 8,
    "replacement_eval_samples": 4,
    "t04_eval_samples": 4,
    "t06_context": 128,
    "calibration_context": 128,
    "t06_ranks": [
      4,
      8
    ]
  },
  "advantage": {
    "rank": 2,
    "key": "lowrank",
    "value": "lowrank",
    "source_mixer": true,
    "rms_calibration": true,
    "bounded_alpha": true
  }
}
```

## IMPLEMENTATION
T04 Ridge Baseline

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
  "task": "T04",
  "n_calib_samples": 8,
  "context_length": 128,
  "mapper_n_params": 7340032,
  "mean_r2": 0.2722043441107848,
  "mean_kv_cosine": 0.7561930237212691,
  "mean_attn_output_cosine": 0.7708631323318087,
  "latency_map_ms_p50": 369.8806249303743,
  "latency_map_ms_p95": 8071.26656792825,
  "repeats": 2,
  "warmup": 1,
  "gate": "PASS",
  "retention_K": 0.7561440673708506,
  "retention_V": 0.7562419800716875,
  "mean_cos_K": 0.7561440673708506,
  "mean_cos_V": 0.7562419800716874,
  "mean_r2_K": 0.27216077007220074,
  "mean_r2_V": 0.2722479181493689,
  "mean_attn_output_cosine_K": 0.7693476813567883,
  "mean_attn_output_cosine_V": 0.772378583306829
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
t05
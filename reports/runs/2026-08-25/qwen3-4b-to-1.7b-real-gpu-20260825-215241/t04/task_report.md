# Task Report

- TASK_ID: `t04`
- STATUS: **PASS**
- Generated at: 2026-08-25T22:09:43.431042

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
T04 Ridge Baseline

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
  "task": "T04",
  "n_calib_samples": 16,
  "context_length": 78,
  "mapper_n_params": 7340032,
  "mean_r2": 0.11384707108729164,
  "mean_kv_cosine": 0.744114871498049,
  "mean_attn_output_cosine": 0.9585075968001089,
  "latency_map_ms_p50": 574.6321952901781,
  "latency_map_ms_p95": 1425.4055598285004,
  "repeats": 10,
  "warmup": 2,
  "data_source": "hf",
  "offline_demo": false,
  "de_rope_k": true,
  "de_rope_v": false,
  "ridge_lambda_k": 0.001,
  "ridge_lambda_v": 0.001,
  "gate": "PASS",
  "retention_K": 0.9625840000286432,
  "retention_V": 0.5256457429674548,
  "mean_cos_K": 0.9625840000286432,
  "mean_cos_V": 0.5256457429674548,
  "mean_r2_K": 0.9214572507739063,
  "mean_r2_V": -0.693763108599323,
  "mean_attn_output_cosine_K": 0.9908602973032399,
  "mean_attn_output_cosine_V": 0.9261548962969778
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
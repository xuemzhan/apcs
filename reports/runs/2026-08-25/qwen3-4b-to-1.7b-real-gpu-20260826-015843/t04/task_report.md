# Task Report

- TASK_ID: `t04`
- STATUS: **PASS**
- Generated at: 2026-08-26T02:01:42.389714

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
    "separate_kv": true,
    "shared_basis": true,
    "source_top_k": 2,
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
  "n_calib_samples": 64,
  "context_length": 70,
  "mapper_n_params": 7340032,
  "mean_r2": -0.00834778650828466,
  "mean_kv_cosine": 0.7716372947759222,
  "mean_attn_output_cosine": 0.9633820977880889,
  "latency_map_ms_p50": 542.6295236684382,
  "latency_map_ms_p95": 2524.44291464052,
  "repeats": 10,
  "warmup": 2,
  "data_source": "hf",
  "offline_demo": false,
  "de_rope_k": true,
  "de_rope_v": false,
  "ridge_lambda_k": 0.001,
  "ridge_lambda_v": 0.001,
  "gate": "PASS",
  "retention_K": 0.9660981994607238,
  "retention_V": 0.5771763900911207,
  "mean_cos_K": 0.9660981994607238,
  "mean_cos_V": 0.5771763900911205,
  "mean_r2_K": 0.9281878212275021,
  "mean_r2_V": -0.9448833942440714,
  "mean_attn_output_cosine_K": 0.9905856620318898,
  "mean_attn_output_cosine_V": 0.9361785335442879
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
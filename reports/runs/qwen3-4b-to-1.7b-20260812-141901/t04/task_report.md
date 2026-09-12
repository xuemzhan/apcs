# Task Report

- TASK_ID: `t04`
- STATUS: **PASS**
- Generated at: 2026-08-12T14:20:25.027659

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
  "mean_r2": 0.27220434411077277,
  "mean_kv_cosine": 0.7561930237213699,
  "mean_attn_output_cosine": 0.7708631323318084,
  "latency_map_ms_p50": 260.1268500002334,
  "latency_map_ms_p95": 3739.1789449946373,
  "repeats": 2,
  "warmup": 1,
  "gate": "PASS",
  "retention_K": 0.7561440673709449,
  "retention_V": 0.7562419800717949,
  "mean_cos_K": 0.7561440673709448,
  "mean_cos_V": 0.7562419800717949,
  "mean_r2_K": 0.27216077007217726,
  "mean_r2_V": 0.2722479181493682,
  "mean_attn_output_cosine_K": 0.7693476813567881,
  "mean_attn_output_cosine_V": 0.7723785833068286
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
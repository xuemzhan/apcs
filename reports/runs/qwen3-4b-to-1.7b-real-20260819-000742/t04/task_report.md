# Task Report

- TASK_ID: `t04`
- STATUS: **PASS**
- Generated at: 2026-08-19T00:46:17.675831

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
- `metadata.json`
- `metrics.json`
- `stdout.log`
- `summary.md`

## KEY_METRICS
```json
{
  "task": "T04",
  "n_calib_samples": 84,
  "context_length": 512,
  "mapper_n_params": 7340032,
  "mean_r2": 0.003402655790361764,
  "mean_kv_cosine": 0.7288645774726099,
  "mean_attn_output_cosine": 0.8103688149774178,
  "latency_map_ms_p50": 1488.1255311192945,
  "latency_map_ms_p95": 1506.347216363065,
  "repeats": 10,
  "warmup": 2,
  "gate": "PASS",
  "retention_K": 0.7288771414505169,
  "retention_V": 0.7288520134947031,
  "mean_cos_K": 0.7288771414505169,
  "mean_cos_V": 0.728852013494703,
  "mean_r2_K": 0.003543105716168271,
  "mean_r2_V": 0.0032622058645552566,
  "mean_attn_output_cosine_K": 0.8100877044153799,
  "mean_attn_output_cosine_V": 0.8106499255394555
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
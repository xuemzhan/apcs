# Task Report

- TASK_ID: `t06`
- STATUS: **OK**
- Generated at: 2026-08-18T23:30:58.552237

## OBJECTIVE
Low-rank 8/16/32 + Shared basis → PCR vs Retention（Figure 1）。

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
T06 Lightweight Mapper

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
  "task": "T06",
  "p_ref_n_params": 7340032,
  "rows": [
    {
      "variant": "lowrank-4",
      "rank": 4,
      "params": 458752,
      "pcr": 0.0625,
      "retention": 0.19898225378924556,
      "r2": 0.022726500637438474,
      "cosine": 0.19898225378924556,
      "retention_K": 0.19830264450869906,
      "retention_V": 0.19966186306979206,
      "r2_K": 0.02270724385951295,
      "r2_V": 0.022745757415364,
      "cosine_K": 0.19830264450869906,
      "cosine_V": 0.19966186306979206
    },
    {
      "variant": "lowrank-8",
      "rank": 8,
      "params": 917504,
      "pcr": 0.125,
      "retention": 0.2582460126215227,
      "r2": 0.03743322788064196,
      "cosine": 0.2582460126215227,
      "retention_K": 0.25661778240077937,
      "retention_V": 0.25987424284226607,
      "r2_K": 0.037167242069301,
      "r2_V": 0.037699213691982925,
      "cosine_K": 0.25661778240077937,
      "cosine_V": 0.25987424284226607
    }
  ]
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
  "gate": "OK"
}

## FAILURE_ANALYSIS
(无)

## NEXT_ALLOWED_TASK
t07
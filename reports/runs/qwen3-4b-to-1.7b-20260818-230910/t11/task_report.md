# Task Report

- TASK_ID: `t11`
- STATUS: **OK**
- Generated at: 2026-08-18T23:45:28.343762

## OBJECTIVE
读 T05/T09/T10 输出 Path A/B/C/D（§69）。

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
T11 MVP Decision

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
  "task": "T11",
  "retention": 0.7561930237212691,
  "chg": 0.16984882992553552,
  "tgrr": 0.5587120464070877,
  "psr_a": 0.0,
  "verdict": "D_STOP_REPLACEABILITY_UNSTABLE",
  "sources": {
    "t05_found": true,
    "t09_found": true,
    "t10_found": false
  }
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
t13
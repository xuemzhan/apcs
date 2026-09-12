# Task Report

- TASK_ID: `t06`
- STATUS: **OK**
- Generated at: 2026-08-19T02:21:00.676498

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
      "variant": "lowrank-8",
      "rank": 8,
      "params": 917504,
      "pcr": 0.125,
      "retention": 0.3178897824520207,
      "r2": 0.03000296333868152,
      "cosine": 0.3178897824520207,
      "retention_K": 0.3176403728681625,
      "retention_V": 0.31813919203587887,
      "r2_K": 0.029918897862062033,
      "r2_V": 0.03008702881530101,
      "cosine_K": 0.31764037286816255,
      "cosine_V": 0.31813919203587887
    },
    {
      "variant": "lowrank-16",
      "rank": 16,
      "params": 1835008,
      "pcr": 0.25,
      "retention": 0.37685054992953126,
      "r2": 0.04277176721486764,
      "cosine": 0.3768505499295314,
      "retention_K": 0.37661337443352155,
      "retention_V": 0.37708772542554103,
      "r2_K": 0.04273072122461896,
      "r2_V": 0.04281281320511632,
      "cosine_K": 0.37661337443352166,
      "cosine_V": 0.37708772542554103
    },
    {
      "variant": "lowrank-32",
      "rank": 32,
      "params": 3670016,
      "pcr": 0.5,
      "retention": 0.36479398961786136,
      "r2": 0.03546119579994422,
      "cosine": 0.36479398961786136,
      "retention_K": 0.363417364775064,
      "retention_V": 0.36617061446065874,
      "r2_K": 0.03515468375337605,
      "r2_V": 0.0357677078465124,
      "cosine_K": 0.36341736477506403,
      "cosine_V": 0.36617061446065874
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
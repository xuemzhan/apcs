# Task Report

- TASK_ID: `t06`
- STATUS: **OK**
- Generated at: 2026-08-25T23:09:27.151908

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
    "rank": 32,
    "separate_kv": true,
    "de_rope": true,
    "de_rope_k": true,
    "de_rope_v": false,
    "ridge_lambda_k": 0.001,
    "ridge_lambda_v": 0.01,
    "real_calibration_samples": 32,
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
T06 Lightweight Mapper

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
  "task": "T06",
  "p_ref_n_params": 7340032,
  "rows": [
    {
      "variant": "lowrank-8",
      "rank": 8,
      "params": 917504,
      "pcr": 0.125,
      "retention": 0.07652014666258337,
      "r2": -0.35103791305397924,
      "cosine": 0.07652014666258337,
      "retention_K": 0.014503698921531609,
      "retention_V": 0.13853659440363514,
      "r2_K": 0.00013221753879943954,
      "r2_V": -0.7022080436467579,
      "cosine_K": 0.014503698921531612,
      "cosine_V": 0.13853659440363514
    },
    {
      "variant": "lowrank-16",
      "rank": 16,
      "params": 1835008,
      "pcr": 0.25,
      "retention": 0.03948925828011644,
      "r2": -0.16824789264169837,
      "cosine": 0.03948925828011644,
      "retention_K": 0.012145842697060536,
      "retention_V": 0.06683267386317235,
      "r2_K": 9.198834755519414e-05,
      "r2_V": -0.33658777363095194,
      "cosine_K": 0.01214584269706054,
      "cosine_V": 0.06683267386317235
    },
    {
      "variant": "lowrank-32",
      "rank": 32,
      "params": 3670016,
      "pcr": 0.5,
      "retention": 0.03120110979671623,
      "r2": -0.18068732426440043,
      "cosine": 0.03120110979671623,
      "retention_K": 0.0073027489964958186,
      "retention_V": 0.05509947059693664,
      "r2_K": 2.4398156519445102e-05,
      "r2_V": -0.3613990466853203,
      "cosine_K": 0.00730274899649582,
      "cosine_V": 0.05509947059693664
    }
  ],
  "data_source": "hf",
  "offline_demo": false,
  "de_rope_k": true,
  "de_rope_v": false
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
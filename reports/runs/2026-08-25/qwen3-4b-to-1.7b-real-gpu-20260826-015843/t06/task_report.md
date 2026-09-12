# Task Report

- TASK_ID: `t06`
- STATUS: **OK**
- Generated at: 2026-08-26T02:19:41.433028

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
      "retention": 0.07015448666438896,
      "r2": -0.3429211785081077,
      "cosine": 0.07015448666438895,
      "retention_K": 0.014458038022059751,
      "retention_V": 0.12585093530671818,
      "r2_K": 0.00013092257084468262,
      "r2_V": -0.6859732795870601,
      "cosine_K": 0.014458038022059751,
      "cosine_V": 0.12585093530671815
    },
    {
      "variant": "lowrank-16",
      "rank": 16,
      "params": 1835008,
      "pcr": 0.25,
      "retention": 0.03340827511445596,
      "r2": -0.12582047870840168,
      "cosine": 0.03340827511445596,
      "retention_K": 0.012272144410476233,
      "retention_V": 0.05454440581843569,
      "r2_K": 9.422024142058394e-05,
      "r2_V": -0.25173517765822395,
      "cosine_K": 0.012272144410476233,
      "cosine_V": 0.054544405818435684
    },
    {
      "variant": "lowrank-32",
      "rank": 32,
      "params": 3670016,
      "pcr": 0.5,
      "retention": 0.02774217002999718,
      "r2": -0.15353757051297767,
      "cosine": 0.027742170029997175,
      "retention_K": 0.007242318300828359,
      "retention_V": 0.048242021759166,
      "r2_K": 2.4379756958348153e-05,
      "r2_V": -0.3070995207829137,
      "cosine_K": 0.007242318300828359,
      "cosine_V": 0.04824202175916599
    },
    {
      "variant": "shared-basis-16",
      "rank": 16,
      "params": 921600,
      "pcr": 0.12555803571428573,
      "retention": 0.6212950480146748,
      "r2": -2.1620096111628397,
      "cosine": 0.6212950480146748,
      "retention_K": 0.9291849209242473,
      "retention_V": 0.31340517510510235,
      "r2_K": 0.8399299826280275,
      "r2_V": -5.163949204953707,
      "cosine_K": 0.9291849209242473,
      "cosine_V": 0.3134051751051023
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
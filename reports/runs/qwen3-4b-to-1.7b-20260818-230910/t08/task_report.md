# Task Report

- TASK_ID: `t08`
- STATUS: **OK**
- Generated at: 2026-08-18T23:38:51.924084

## OBJECTIVE
Source-Layer Mixer + K/V 独立低秩残差 + RMS Calibration + Bounded α。

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
T08 Advantage State Training

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
  "task": "T08",
  "rank": 2,
  "alpha_max": 0.5,
  "alpha_layer_mean": 0.0,
  "alpha_layer_std": 0.0,
  "base_rms": 0.5667192511858541,
  "residual_K_rms": 0.09139494157526558,
  "residual_V_rms": 0.092523358981699,
  "ratio_K": 0.16127022574938582,
  "ratio_V": 0.1632613658140161,
  "loss_init": 2.1601270768279353,
  "loss_final": 1.3168650511286666,
  "n_loss": 100,
  "loss_init_K": 2.1730401567276183,
  "loss_final_K": 1.3169102005338515,
  "loss_init_V": 2.147213996928252,
  "loss_final_V": 1.3168199017234818,
  "n_layers_teacher": 36,
  "n_layers_student": 28,
  "student_frozen": true,
  "ablation_switches": {
    "A1_rank_fixed": 16,
    "A9_rms_calibration": true,
    "A10_bounded_alpha": true,
    "A8_separate_kv": true
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
t09
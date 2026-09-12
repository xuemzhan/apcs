# Task Report

- TASK_ID: `t03`
- STATUS: **OK**
- Generated at: 2026-08-25T23:28:23.724701

## OBJECTIVE
构造 Teacher↔Student 4 种层映射策略：proportional / last / data-driven / geometry-aware。

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
    "rms_calibration": true,
    "separate_kv": false,
    "shared_basis": false,
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
T03 Layer Alignment

## OUTPUT_FILES
- `compliance.json`
- `config.json`
- `layer_mapping.json`
- `metadata.json`
- `metrics.json`
- `provider.json`
- `stdout.log`
- `summary.md`

## KEY_METRICS
```json
{
  "task": "T03",
  "n_teacher_layers": 36,
  "n_student_layers": 28,
  "strategy": [
    "proportional",
    "last_layer",
    "data_driven_topk",
    "geometry_aware_topk"
  ],
  "mapping_size_avg": {
    "proportional": 1.2857142857142858,
    "last_layer": 2.0,
    "data_driven_topk": 2.0,
    "geometry_aware_topk": 2.0
  },
  "selected": "proportional",
  "note": "offline demo 用对角带状合成相似度矩阵；真实实验需 attn-output cosine。"
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
t04
# Task Report

- TASK_ID: `prepare-data`
- STATUS: **OK**
- Generated at: 2026-08-31T00:35:32.585759

## OBJECTIVE
下载并规范化真实数据，冻结互斥 train/validation/test manifest。

## MODEL_PAIR
- Teacher: `Qwen/Qwen3-4b`
- Student: `Qwen/Qwen3-1.7b`

## DATASET
mmlu

## CONFIG
```json
{
  "context_lengths": [
    512
  ],
  "seeds": [
    42
  ],
  "mapper": {
    "type": "affine",
    "separate_kv": true,
    "de_rope": true,
    "de_rope_k": true,
    "de_rope_v": false,
    "ridge_lambda_k": 0.001,
    "ridge_lambda_v": 0.001,
    "rank": 16,
    "inject_eval_calib_samples": 30,
    "layer_selection": "proportional",
    "inject_eval": true,
    "rope_align": "rotated"
  },
  "advantage": null
}
```

## IMPLEMENTATION
Dataset Preparation

## OUTPUT_FILES
- `compliance.json`
- `config.json`
- `dataset_manifest.json`
- `dataset_samples.json`
- `metadata.json`
- `metrics.json`
- `provider.json`
- `stdout.log`
- `summary.md`

## KEY_METRICS
```json
{
  "task": "prepare-data",
  "datasets": [
    "hellaswag",
    "arc_challenge",
    "mmlu"
  ],
  "seed": 42,
  "split_overlap_check": "PASS",
  "manifest": "dataset_manifest.json",
  "snapshot": "dataset_samples.json"
}
```

## STATISTICAL_CHECK
{
  "n_samples": null,
  "seeds": [
    42
  ],
  "ci": null
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
(全部完成)
# Task Report

- TASK_ID: `t01`
- STATUS: **PASS**
- Generated at: 2026-08-25T23:39:43.527872

## OBJECTIVE
验证 Student 自产 KV 注入与原生推理等价（Gate 0）。

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
T01 Self-KV Replay

## OUTPUT_FILES
- `compliance.json`
- `config.json`
- `dataset_manifest.json`
- `metadata.json`
- `metrics.json`
- `provider.json`
- `stdout.log`
- `summary.md`

## KEY_METRICS
```json
{
  "task": "T01",
  "n_samples": 4,
  "mean_logit_cosine": 1.0,
  "mean_max_error": 0.0,
  "mean_token_agreement": 1.0,
  "gate0": "PASS",
  "offline_demo": false,
  "note": "T01 为真实 GPU Self-KV Replay：保留 HF 原生 cache 作为对照，与 numpy offload → cache 重建路径逐步比较 logits 与 token（§29）。offline_demo=False，可作 Engineering Correctness 证据。"
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
t02
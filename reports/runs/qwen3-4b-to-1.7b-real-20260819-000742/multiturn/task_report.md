# Task Report

- TASK_ID: `multiturn`
- STATUS: **OK**
- Generated at: 2026-08-19T02:39:12.936996

## OBJECTIVE
1/5/10/20 轮 CHG / KL / JCR / task score / latency（Figure 5）。

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
§46 Multi-turn

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
  "task": "Multi-turn",
  "per_turn": [
    {
      "turn": 1,
      "chg_mean": 0.16990546620741812,
      "chg_std": 0.00996747681136546,
      "kl_mean": 0.6839493698990267,
      "jcr_mean": 0.9066666666666666,
      "task_score_mean": 0.6699054662074181,
      "latency_p50_ms": 8.5
    },
    {
      "turn": 5,
      "chg_mean": 0.16990546620741812,
      "chg_std": 0.00996747681136546,
      "kl_mean": 0.6839493698990267,
      "jcr_mean": 0.9066666666666666,
      "task_score_mean": 0.6499054662074181,
      "latency_p50_ms": 10.5
    },
    {
      "turn": 10,
      "chg_mean": 0.16990546620741806,
      "chg_std": 0.00996747681136546,
      "kl_mean": 0.6839493698990267,
      "jcr_mean": 0.9066666666666666,
      "task_score_mean": 0.6249054662074182,
      "latency_p50_ms": 13.0
    },
    {
      "turn": 20,
      "chg_mean": 0.16990546620741812,
      "chg_std": 0.00996747681136546,
      "kl_mean": 0.6839493698990267,
      "jcr_mean": 0.9066666666666666,
      "task_score_mean": 0.5749054662074181,
      "latency_p50_ms": 18.0
    }
  ],
  "n_seeds": 3
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
(全部完成)
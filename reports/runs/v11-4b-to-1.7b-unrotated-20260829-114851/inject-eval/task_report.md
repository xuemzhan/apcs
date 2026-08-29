# Task Report

- TASK_ID: `inject-eval`
- STATUS: **FAIL**
- Generated at: 2026-08-29T11:54:50.107112

## OBJECTIVE
Real path: Teacher prefill → map → inject → Student zero-prefill scoring.

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
    "type": "ridge",
    "separate_kv": true,
    "de_rope": true,
    "de_rope_k": true,
    "de_rope_v": false,
    "ridge_lambda_k": 0.001,
    "ridge_lambda_v": 0.001,
    "inject_eval_calib_samples": 20,
    "layer_selection": "proportional",
    "inject_eval": true,
    "rope_align": "unrotated"
  },
  "advantage": null
}
```

## IMPLEMENTATION
Inject Eval

## OUTPUT_FILES
- `compliance.json`
- `config.json`
- `metadata.json`
- `metrics.json`
- `provider.json`
- `stdout.log`
- `summary.md`

## KEY_METRICS
```json
{
  "n_samples": 20,
  "n_capability_records": 100,
  "zero_prefill_verified": false,
  "protocol_version": "1.1",
  "git_hash": "f3051e1dba5b99c3dd2881e5d3162c0959a3e391",
  "online": false,
  "calib_eval_disjoint": true,
  "rope_align": "unrotated",
  "method_stats": {
    "ridge_kv_both": {
      "total": 20,
      "accuracy": 0.1,
      "gold_prob_mean": null,
      "conf_mean": 0.0,
      "ppl_mean": null
    },
    "ridge_native": {
      "total": 20,
      "accuracy": 0.1,
      "gold_prob_mean": null,
      "conf_mean": 0.0,
      "ppl_mean": null
    },
    "student": {
      "total": 20,
      "accuracy": 0.4,
      "gold_prob_mean": 0.42111083782598185,
      "conf_mean": 0.7220165922569729,
      "ppl_mean": 9.539551374377966
    },
    "teacher": {
      "total": 20,
      "accuracy": 0.7,
      "gold_prob_mean": 0.6283512442536174,
      "conf_mean": 0.816891959778712,
      "ppl_mean": null
    },
    "text": {
      "total": 20,
      "accuracy": 0.4,
      "gold_prob_mean": 0.4230656632873927,
      "conf_mean": 0.7260926489616222,
      "ppl_mean": null
    }
  },
  "chg_gold": {
    "ridge_kv_both": {},
    "ridge_native": {}
  },
  "chg_accuracy": {
    "ridge_kv_both": {
      "mean": -0.3,
      "ci_low": -0.55,
      "ci_high": 0.0,
      "p": 0.99000999000999,
      "n": 20
    },
    "ridge_native": {
      "mean": -0.3,
      "ci_low": -0.55,
      "ci_high": 0.0,
      "p": 0.99000999000999,
      "n": 20
    }
  },
  "capability_gate": {
    "metric": "gold_prob",
    "method": "ridge_kv_both",
    "chg": null,
    "ci_low": null,
    "ci_high": null,
    "p": null,
    "gate": "FAIL"
  },
  "psr": {
    "mean": 1.0,
    "median": 1.0,
    "n": 20,
    "note": "t_handoff 含测量专用 PPL forward（部署时不需要），PSR 偏保守；校准阶段 student self-prefill 成本未计入（见 mapper 校准成本）。"
  },
  "method_means": {
    "ridge_kv_both": 0.0,
    "ridge_native": 0.0,
    "student": 0.7220165922569729,
    "teacher": 0.816891959778712,
    "text": 0.7260926489616222
  },
  "ppl_means": {
    "student": 9.539551374377966
  },
  "chg": -0.7220165922569729,
  "ridge_kv_both_score_mean": 0.0,
  "student_score_mean": 0.7220165922569729
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
  "gate": "FAIL"
}

## FAILURE_ANALYSIS
(无)

## NEXT_ALLOWED_TASK
(全部完成)
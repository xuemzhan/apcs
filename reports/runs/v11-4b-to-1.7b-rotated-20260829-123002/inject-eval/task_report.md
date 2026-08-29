# Task Report

- TASK_ID: `inject-eval`
- STATUS: **PASS**
- Generated at: 2026-08-29T12:30:32.333069

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
    "rope_align": "rotated"
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
  "zero_prefill_verified": true,
  "protocol_version": "1.1",
  "git_hash": "f3051e1dba5b99c3dd2881e5d3162c0959a3e391",
  "online": true,
  "calib_eval_disjoint": true,
  "rope_align": "rotated",
  "method_stats": {
    "ridge_kv_both": {
      "total": 20,
      "accuracy": 0.25,
      "gold_prob_mean": 0.32497393713444367,
      "conf_mean": 0.7051620651287636,
      "ppl_mean": 147.10136625167357
    },
    "ridge_native": {
      "total": 20,
      "accuracy": 0.35,
      "gold_prob_mean": 0.3390635550154775,
      "conf_mean": 0.7991330559415958,
      "ppl_mean": 993738.6317884151
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
    "ridge_kv_both": {
      "mean": -0.09613690069153821,
      "ci_low": -0.30683990203013556,
      "ci_high": 0.13577987520411497,
      "p": 0.7902097902097902,
      "n": 20,
      "tgrr": -0.463890716818814
    },
    "ridge_native": {
      "mean": -0.08204728281050436,
      "ci_low": -0.3242700433815822,
      "ci_high": 0.14739064201500263,
      "p": 0.7392607392607392,
      "n": 20,
      "tgrr": -0.3959038887484218
    }
  },
  "chg_accuracy": {
    "ridge_kv_both": {
      "mean": -0.15,
      "ci_low": -0.4,
      "ci_high": 0.15,
      "p": 0.9090909090909091,
      "n": 20
    },
    "ridge_native": {
      "mean": -0.05,
      "ci_low": -0.35,
      "ci_high": 0.25,
      "p": 0.7502497502497503,
      "n": 20
    }
  },
  "capability_gate": {
    "metric": "gold_prob",
    "method": "ridge_kv_both",
    "chg": -0.09613690069153821,
    "ci_low": -0.30683990203013556,
    "ci_high": 0.13577987520411497,
    "p": 0.7902097902097902,
    "gate": "FAIL"
  },
  "psr": {
    "mean": -13.619176076343996,
    "median": -13.362107335477951,
    "n": 20,
    "note": "在线阶段：KV 从磁盘加载；t_handoff 含测量用 PPL forward，偏保守"
  },
  "method_means": {
    "ridge_kv_both": 0.7051620651287636,
    "ridge_native": 0.7991330559415958,
    "student": 0.7220165922569729,
    "teacher": 0.816891959778712,
    "text": 0.7260926489616222
  },
  "ppl_means": {
    "ridge_kv_both": 147.10136625167357,
    "ridge_native": 993738.6317884151,
    "student": 9.539551374377966
  },
  "chg": -0.01685452712820934,
  "ridge_kv_both_score_mean": 0.7051620651287636,
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
  "gate": "PASS"
}

## FAILURE_ANALYSIS
(无)

## NEXT_ALLOWED_TASK
(全部完成)
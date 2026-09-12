# Task Report

- TASK_ID: `inject-eval`
- STATUS: **PASS**
- Generated at: 2026-09-12T20:14:42.618960

## OBJECTIVE
Real path: Teacher prefill → map → inject → Student zero-prefill scoring.

## MODEL_PAIR
- Teacher: `/workspace/models/Qwen3-4B`
- Student: `/workspace/models/Qwen3-1.7B`

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
  "n_samples": 30,
  "n_capability_records": 180,
  "zero_prefill_verified": true,
  "protocol_version": "1.1",
  "git_hash": "4e99fa58a52ec9506a17050ac19bf3ef75089178",
  "online": true,
  "calib_eval_disjoint": true,
  "rope_align": "rotated",
  "method_stats": {
    "ridge_kv_both": {
      "total": 30,
      "accuracy": 0.36666666666666664,
      "gold_prob_mean": 0.3653994358476148,
      "conf_mean": 0.6742900763958091,
      "ppl_mean": 46.08202474920152,
      "logit_cos_mean": 0.533020887878854
    },
    "ridge_native": {
      "total": 30,
      "accuracy": 0.26666666666666666,
      "gold_prob_mean": 0.2640457519172628,
      "conf_mean": 0.8215618708341744,
      "ppl_mean": 3334889.560767581,
      "logit_cos_mean": 0.1910610432451812
    },
    "ridge_self_kv": {
      "total": 30,
      "accuracy": 0.5,
      "gold_prob_mean": 0.5030914978382872,
      "conf_mean": 0.856821331486245,
      "ppl_mean": 21.226648495205207,
      "logit_cos_mean": 0.9999805623717897
    },
    "student": {
      "total": 30,
      "accuracy": 0.5,
      "gold_prob_mean": 0.5029589363646234,
      "conf_mean": 0.8572382295459581,
      "ppl_mean": 25.35546443570249,
      "logit_cos_mean": null
    },
    "teacher": {
      "total": 30,
      "accuracy": 0.7666666666666667,
      "gold_prob_mean": 0.6934298316020462,
      "conf_mean": 0.8430357019480165,
      "ppl_mean": null,
      "logit_cos_mean": null
    },
    "text": {
      "total": 30,
      "accuracy": 0.5,
      "gold_prob_mean": 0.5029589363646234,
      "conf_mean": 0.8572382295459581,
      "ppl_mean": null,
      "logit_cos_mean": null
    }
  },
  "chg_gold": {
    "ridge_kv_both": {
      "mean": -0.1375595005170087,
      "ci_low": -0.3174873808880432,
      "ci_high": 0.029496943961391273,
      "p": 0.9150849150849151,
      "n": 30,
      "tgrr": -0.7222074550840967
    },
    "ridge_native": {
      "mean": -0.23891318444736068,
      "ci_low": -0.44279053103815175,
      "ci_high": -0.03752805543364396,
      "p": 0.987012987012987,
      "n": 30,
      "tgrr": -1.2543290887017378
    },
    "ridge_self_kv": {
      "mean": 0.00013256147366369635,
      "ci_low": -0.0006186238273156126,
      "ci_high": 0.0009351698490712134,
      "p": 0.39760239760239763,
      "n": 30,
      "tgrr": 0.0006959670846218154
    }
  },
  "chg_accuracy": {
    "ridge_kv_both": {
      "mean": -0.13333333333333333,
      "ci_low": -0.3333333333333333,
      "ci_high": 0.1,
      "p": 0.922077922077922,
      "n": 30
    },
    "ridge_native": {
      "mean": -0.23333333333333334,
      "ci_low": -0.4666666666666667,
      "ci_high": 0.0,
      "p": 0.9820179820179821,
      "n": 30
    },
    "ridge_self_kv": {
      "mean": 0.0,
      "ci_low": 0.0,
      "ci_high": 0.0,
      "p": 1.0,
      "n": 30
    }
  },
  "capability_gate": {
    "metric": "gold_prob",
    "method": "ridge_kv_both",
    "chg": -0.1375595005170087,
    "ci_low": -0.3174873808880432,
    "ci_high": 0.029496943961391273,
    "p": 0.9150849150849151,
    "gate": "FAIL"
  },
  "psr": {
    "mean": -12.120831371939511,
    "median": -11.653965156205338,
    "n": 30,
    "note": "在线阶段：KV 从磁盘加载；t_handoff 含测量用 PPL forward，偏保守"
  },
  "kv_norm_diagnostics": {},
  "method_means": {
    "ridge_kv_both": 0.6742900763958091,
    "ridge_native": 0.8215618708341744,
    "ridge_self_kv": 0.856821331486245,
    "student": 0.8572382295459581,
    "teacher": 0.8430357019480165,
    "text": 0.8572382295459581
  },
  "ppl_means": {
    "ridge_kv_both": 46.08202474920152,
    "ridge_native": 3334889.560767581,
    "ridge_self_kv": 21.226648495205207,
    "student": 25.35546443570249
  },
  "chg": -0.182948153150149,
  "ridge_kv_both_score_mean": 0.6742900763958091,
  "student_score_mean": 0.8572382295459581
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
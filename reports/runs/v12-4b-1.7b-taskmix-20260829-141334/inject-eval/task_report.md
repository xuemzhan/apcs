# Task Report

- TASK_ID: `inject-eval`
- STATUS: **PASS**
- Generated at: 2026-08-29T14:16:02.622523

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
    "type": "task_aware",
    "separate_kv": true,
    "de_rope": true,
    "de_rope_k": true,
    "de_rope_v": false,
    "ridge_lambda_k": 0.001,
    "ridge_lambda_v": 0.001,
    "inject_eval_calib_samples": 30,
    "layer_selection": "proportional",
    "inject_eval": true,
    "rope_align": "rotated",
    "task_loss_weight": 0.7,
    "task_delta_mode": "diag",
    "learn_layer_alpha": true,
    "task_lr": 0.003,
    "task_steps": 40,
    "task_batch_size": 4
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
  "n_capability_records": 210,
  "zero_prefill_verified": true,
  "protocol_version": "1.1",
  "git_hash": "f3051e1dba5b99c3dd2881e5d3162c0959a3e391",
  "online": false,
  "calib_eval_disjoint": true,
  "rope_align": "rotated",
  "method_stats": {
    "ridge_kv_both": {
      "total": 30,
      "accuracy": 0.2,
      "gold_prob_mean": 0.20692027222643436,
      "conf_mean": 0.7227497449594552,
      "ppl_mean": 88.46260485304286,
      "logit_cos_mean": 0.5564752920621612
    },
    "ridge_native": {
      "total": 30,
      "accuracy": 0.26666666666666666,
      "gold_prob_mean": 0.2640457519172628,
      "conf_mean": 0.8215618708341749,
      "ppl_mean": 3334889.560767582,
      "logit_cos_mean": 0.19106104324518128
    },
    "ridge_self_kv": {
      "total": 30,
      "accuracy": 0.5,
      "gold_prob_mean": 0.5030914978382872,
      "conf_mean": 0.856821331486245,
      "ppl_mean": 21.226648495205207,
      "logit_cos_mean": 0.9999805623717901
    },
    "student": {
      "total": 30,
      "accuracy": 0.5,
      "gold_prob_mean": 0.5029589363646235,
      "conf_mean": 0.857238229545958,
      "ppl_mean": 25.35546443570249,
      "logit_cos_mean": null
    },
    "summary": {
      "total": 30,
      "accuracy": 0.36666666666666664,
      "gold_prob_mean": 0.37172481970639637,
      "conf_mean": 0.7381563420372801,
      "ppl_mean": null,
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
      "gold_prob_mean": 0.5029589363646235,
      "conf_mean": 0.857238229545958,
      "ppl_mean": null,
      "logit_cos_mean": null
    }
  },
  "chg_gold": {
    "ridge_kv_both": {
      "mean": -0.29603866413818913,
      "ci_low": -0.47919495150969627,
      "ci_high": -0.12105463590544983,
      "p": 0.999000999000999,
      "n": 30,
      "tgrr": -1.5542461947751958
    },
    "ridge_native": {
      "mean": -0.23891318444736068,
      "ci_low": -0.44279053103815175,
      "ci_high": -0.03752805543364396,
      "p": 0.987012987012987,
      "n": 30,
      "tgrr": -1.2543290887017384
    },
    "ridge_self_kv": {
      "mean": 0.00013256147366369635,
      "ci_low": -0.0006186238273156126,
      "ci_high": 0.0009351698490712134,
      "p": 0.39760239760239763,
      "n": 30,
      "tgrr": 0.0006959670846218158
    }
  },
  "chg_accuracy": {
    "ridge_kv_both": {
      "mean": -0.3,
      "ci_low": -0.5333333333333333,
      "ci_high": -0.06583333333333409,
      "p": 0.9920079920079921,
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
    "chg": -0.29603866413818913,
    "ci_low": -0.47919495150969627,
    "ci_high": -0.12105463590544983,
    "p": 0.999000999000999,
    "gate": "FAIL"
  },
  "psr": {
    "mean": -3.7267166740299964,
    "median": -3.9453166891984495,
    "n": 30,
    "note": "t_handoff 含测量专用 PPL forward（部署时不需要），PSR 偏保守；校准阶段 student self-prefill 成本未计入（见 mapper 校准成本）。"
  },
  "kv_norm_diagnostics": {},
  "method_means": {
    "ridge_kv_both": 0.7227497449594552,
    "ridge_native": 0.8215618708341749,
    "ridge_self_kv": 0.856821331486245,
    "student": 0.857238229545958,
    "summary": 0.7381563420372801,
    "teacher": 0.8430357019480165,
    "text": 0.857238229545958
  },
  "ppl_means": {
    "ridge_kv_both": 88.46260485304286,
    "ridge_native": 3334889.560767582,
    "ridge_self_kv": 21.226648495205207,
    "student": 25.35546443570249
  },
  "chg": -0.13448848458650275,
  "ridge_kv_both_score_mean": 0.7227497449594552,
  "student_score_mean": 0.857238229545958
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
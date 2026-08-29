# Task Report

- TASK_ID: `inject-eval`
- STATUS: **PASS**
- Generated at: 2026-08-29T14:13:33.440284

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
    "type": "affine",
    "separate_kv": true,
    "de_rope": true,
    "de_rope_k": true,
    "de_rope_v": false,
    "ridge_lambda_k": 0.001,
    "ridge_lambda_v": 0.001,
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
  "n_capability_records": 270,
  "zero_prefill_verified": true,
  "protocol_version": "1.1",
  "git_hash": "f3051e1dba5b99c3dd2881e5d3162c0959a3e391",
  "online": false,
  "calib_eval_disjoint": true,
  "rope_align": "rotated",
  "method_stats": {
    "ridge_k_only": {
      "total": 30,
      "accuracy": 0.26666666666666666,
      "gold_prob_mean": 0.21789697404658245,
      "conf_mean": 0.686859632168714,
      "ppl_mean": 148.52501772974813,
      "logit_cos_mean": 0.5602689603893599
    },
    "ridge_kv_both": {
      "total": 30,
      "accuracy": 0.36666666666666664,
      "gold_prob_mean": 0.36539943584761486,
      "conf_mean": 0.674290076395809,
      "ppl_mean": 46.08202474920153,
      "logit_cos_mean": 0.5330208878788539
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
    "ridge_v_only": {
      "total": 30,
      "accuracy": 0.3333333333333333,
      "gold_prob_mean": 0.35085837874040077,
      "conf_mean": 0.8282070831906786,
      "ppl_mean": 14557744.630471429,
      "logit_cos_mean": 0.030985467098015303
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
    "ridge_k_only": {
      "mean": -0.2850619623180411,
      "ci_low": -0.44497053496586164,
      "ci_high": -0.14130183355618608,
      "p": 0.999000999000999,
      "n": 30,
      "tgrr": -1.4966169081250462
    },
    "ridge_kv_both": {
      "mean": -0.1375595005170087,
      "ci_low": -0.3174873808880432,
      "ci_high": 0.029496943961391273,
      "p": 0.9150849150849151,
      "n": 30,
      "tgrr": -0.7222074550840971
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
    },
    "ridge_v_only": {
      "mean": -0.15210055762422275,
      "ci_low": -0.35604372828951236,
      "ci_high": 0.05174919486949386,
      "p": 0.9050949050949051,
      "n": 30,
      "tgrr": -0.7985501272235259
    }
  },
  "chg_accuracy": {
    "ridge_k_only": {
      "mean": -0.23333333333333334,
      "ci_low": -0.4666666666666667,
      "ci_high": 0.0,
      "p": 0.981018981018981,
      "n": 30
    },
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
    },
    "ridge_v_only": {
      "mean": -0.16666666666666666,
      "ci_low": -0.4008333333333326,
      "ci_high": 0.1,
      "p": 0.9280719280719281,
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
    "mean": -10.755893365358366,
    "median": -9.851180110075312,
    "n": 30,
    "note": "t_handoff 含测量专用 PPL forward（部署时不需要），PSR 偏保守；校准阶段 student self-prefill 成本未计入（见 mapper 校准成本）。"
  },
  "kv_norm_diagnostics": {},
  "method_means": {
    "ridge_k_only": 0.686859632168714,
    "ridge_kv_both": 0.674290076395809,
    "ridge_native": 0.8215618708341749,
    "ridge_self_kv": 0.856821331486245,
    "ridge_v_only": 0.8282070831906786,
    "student": 0.857238229545958,
    "summary": 0.7381563420372801,
    "teacher": 0.8430357019480165,
    "text": 0.857238229545958
  },
  "ppl_means": {
    "ridge_k_only": 148.52501772974813,
    "ridge_kv_both": 46.08202474920153,
    "ridge_native": 3334889.560767582,
    "ridge_self_kv": 21.226648495205207,
    "ridge_v_only": 14557744.630471429,
    "student": 25.35546443570249
  },
  "chg": -0.182948153150149,
  "ridge_kv_both_score_mean": 0.674290076395809,
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
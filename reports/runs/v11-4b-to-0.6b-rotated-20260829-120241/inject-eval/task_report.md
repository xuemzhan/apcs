# Task Report

- TASK_ID: `inject-eval`
- STATUS: **PASS**
- Generated at: 2026-08-29T12:06:07.933399

## OBJECTIVE
Real path: Teacher prefill → map → inject → Student zero-prefill scoring.

## MODEL_PAIR
- Teacher: `Qwen/Qwen3-4b`
- Student: `Qwen/Qwen3-0.6b`

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
  "online": false,
  "calib_eval_disjoint": true,
  "rope_align": "rotated",
  "method_stats": {
    "ridge_kv_both": {
      "total": 20,
      "accuracy": 0.15,
      "gold_prob_mean": 0.21925283206572685,
      "conf_mean": 0.74457424190934,
      "ppl_mean": 141.84218079037367
    },
    "ridge_native": {
      "total": 20,
      "accuracy": 0.2,
      "gold_prob_mean": 0.2381666233580654,
      "conf_mean": 0.8094280098703021,
      "ppl_mean": 14763.286419676268
    },
    "student": {
      "total": 20,
      "accuracy": 0.1,
      "gold_prob_mean": 0.20877190028502263,
      "conf_mean": 0.5586530667496985,
      "ppl_mean": 9.899930128151857
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
      "accuracy": 0.1,
      "gold_prob_mean": 0.2083039236946053,
      "conf_mean": 0.5560473772832951,
      "ppl_mean": null
    }
  },
  "chg_gold": {
    "ridge_kv_both": {
      "mean": 0.010480931780704288,
      "ci_low": -0.1268343633427243,
      "ci_high": 0.1583832373574453,
      "p": 0.4125874125874126,
      "n": 20,
      "tgrr": 0.024979618113633303
    },
    "ridge_native": {
      "mean": 0.029394723073042788,
      "ci_low": -0.1052151999161026,
      "ci_high": 0.19385029679532165,
      "p": 0.3786213786213786,
      "n": 20,
      "tgrr": 0.07005760291965889
    }
  },
  "chg_accuracy": {
    "ridge_kv_both": {
      "mean": 0.05,
      "ci_low": -0.15,
      "ci_high": 0.25,
      "p": 0.48151848151848153,
      "n": 20
    },
    "ridge_native": {
      "mean": 0.1,
      "ci_low": -0.15,
      "ci_high": 0.35,
      "p": 0.3336663336663337,
      "n": 20
    }
  },
  "capability_gate": {
    "metric": "gold_prob",
    "method": "ridge_kv_both",
    "chg": 0.010480931780704288,
    "ci_low": -0.1268343633427243,
    "ci_high": 0.1583832373574453,
    "p": 0.4125874125874126,
    "gate": "FAIL"
  },
  "psr": {
    "mean": -13.636125590805682,
    "median": -13.064591080367991,
    "n": 20,
    "note": "t_handoff 含测量专用 PPL forward（部署时不需要），PSR 偏保守；校准阶段 student self-prefill 成本未计入（见 mapper 校准成本）。"
  },
  "method_means": {
    "ridge_kv_both": 0.74457424190934,
    "ridge_native": 0.8094280098703021,
    "student": 0.5586530667496985,
    "teacher": 0.816891959778712,
    "text": 0.5560473772832951
  },
  "ppl_means": {
    "ridge_kv_both": 141.84218079037367,
    "ridge_native": 14763.286419676268,
    "student": 9.899930128151857
  },
  "chg": 0.18592117515964146,
  "ridge_kv_both_score_mean": 0.74457424190934,
  "student_score_mean": 0.5586530667496985
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
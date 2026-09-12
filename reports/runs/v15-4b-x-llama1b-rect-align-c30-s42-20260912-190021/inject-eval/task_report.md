# Task Report

- TASK_ID: `inject-eval`
- STATUS: **PASS**
- Generated at: 2026-09-12T19:01:43.125436

## OBJECTIVE
Real path: Teacher prefill → map → inject → Student zero-prefill scoring.

## MODEL_PAIR
- Teacher: `/workspace/models/Qwen3-4B`
- Student: `/workspace/models/Llama-3.2-1B`

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
    "de_rope": true,
    "de_rope_k": true,
    "de_rope_v": false,
    "inject_eval": true,
    "inject_eval_calib_samples": 30,
    "layer_selection": "proportional",
    "rank": 16,
    "ridge_lambda_k": 0.001,
    "ridge_lambda_v": 0.001,
    "rope_align": "rotated",
    "separate_kv": true,
    "type": "rect_affine"
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
  "n_samples": 100,
  "n_capability_records": 500,
  "zero_prefill_verified": true,
  "protocol_version": "1.1",
  "git_hash": "4e99fa58a52ec9506a17050ac19bf3ef75089178",
  "online": false,
  "calib_eval_disjoint": true,
  "rope_align": "rotated",
  "method_stats": {
    "ridge_kv_both": {
      "total": 100,
      "accuracy": 0.31,
      "gold_prob_mean": 0.2514757048458647,
      "conf_mean": 0.3123431393866423,
      "ppl_mean": 29.0456752388901,
      "logit_cos_mean": 0.9922712895763542
    },
    "ridge_self_kv": {
      "total": 100,
      "accuracy": 0.29,
      "gold_prob_mean": 0.2544270947520587,
      "conf_mean": 0.32821237488919375,
      "ppl_mean": 23.136744193421382,
      "logit_cos_mean": 0.9999986995771593
    },
    "student": {
      "total": 100,
      "accuracy": 0.3,
      "gold_prob_mean": 0.2544229668780418,
      "conf_mean": 0.3282495887884593,
      "ppl_mean": 30.79558272604409,
      "logit_cos_mean": null
    },
    "teacher": {
      "total": 100,
      "accuracy": 0.77,
      "gold_prob_mean": 0.6907188163880084,
      "conf_mean": 0.807396388985802,
      "ppl_mean": null,
      "logit_cos_mean": null
    },
    "text": {
      "total": 100,
      "accuracy": 0.3,
      "gold_prob_mean": 0.2544229668780418,
      "conf_mean": 0.3282495887884593,
      "ppl_mean": null,
      "logit_cos_mean": null
    }
  },
  "chg_gold": {
    "ridge_kv_both": {
      "mean": -0.002947262032177108,
      "ci_low": -0.011160447274509376,
      "ci_high": 0.005294449691508141,
      "p": 0.7582417582417582,
      "n": 100,
      "tgrr": -0.00675519154144459
    },
    "ridge_self_kv": {
      "mean": 4.127874016903454e-06,
      "ci_low": -0.00016114154823139781,
      "ci_high": 0.00017975874026789312,
      "p": 0.4805194805194805,
      "n": 100,
      "tgrr": 9.461181034703285e-06
    }
  },
  "chg_accuracy": {
    "ridge_kv_both": {
      "mean": 0.01,
      "ci_low": -0.08,
      "ci_high": 0.1,
      "p": 0.4995004995004995,
      "n": 100
    },
    "ridge_self_kv": {
      "mean": -0.01,
      "ci_low": -0.03,
      "ci_high": 0.0,
      "p": 1.0,
      "n": 100
    }
  },
  "capability_gate": {
    "metric": "gold_prob",
    "method": "ridge_kv_both",
    "chg": -0.002947262032177108,
    "ci_low": -0.011160447274509376,
    "ci_high": 0.005294449691508141,
    "p": 0.7582417582417582,
    "gate": "FAIL"
  },
  "psr": {
    "mean": -9.786892589279597,
    "median": -9.713471977554612,
    "n": 100,
    "note": "t_handoff 含测量专用 PPL forward（部署时不需要），PSR 偏保守；校准阶段 student self-prefill 成本未计入（见 mapper 校准成本）。"
  },
  "kv_norm_diagnostics": {
    "K": {
      "teacher_per_layer": [
        148.8058624267578,
        33.415626525878906,
        18.555912017822266,
        22.429798126220703,
        32.67997741699219,
        50.15192794799805,
        26.124723434448242,
        21.616235733032227,
        33.03757858276367,
        20.31751823425293,
        28.626596450805664,
        28.838281631469727,
        23.93501091003418,
        20.942964553833008,
        25.848825454711914,
        22.055944442749023,
        20.87969398498535,
        22.1553955078125,
        21.980815887451172,
        22.799299240112305,
        22.157459259033203,
        22.818634033203125,
        23.430065155029297,
        26.196834564208984,
        22.18714141845703,
        23.84137725830078,
        20.358551025390625,
        19.75421905517578,
        20.500396728515625,
        21.16212272644043,
        22.949621200561523,
        21.39583396911621,
        20.9459171295166,
        20.402727127075195,
        21.233501434326172,
        30.566564559936523
      ],
      "student_per_layer": [
        14.716822624206543,
        18.62373161315918,
        17.341238021850586,
        18.95745086669922,
        18.57101821899414,
        19.864381790161133,
        19.917560577392578,
        16.702762603759766,
        18.592174530029297,
        18.774168014526367,
        17.380725860595703,
        20.088565826416016,
        18.214494705200195,
        17.593292236328125,
        15.611028671264648,
        17.21609115600586
      ],
      "ratio_mean": 0.6694697141647339
    },
    "V": {
      "teacher_per_layer": [
        0.34937870502471924,
        0.5673103332519531,
        0.7894033789634705,
        1.0674461126327515,
        1.3695234060287476,
        1.5490434169769287,
        1.988561749458313,
        2.9348721504211426,
        3.81805682182312,
        3.9553756713867188,
        5.153398036956787,
        3.5625946521759033,
        3.8847248554229736,
        3.809962272644043,
        4.481847286224365,
        4.748594284057617,
        6.110572338104248,
        5.441842555999756,
        6.124639511108398,
        6.31256628036499,
        6.1214704513549805,
        6.909948825836182,
        8.512569427490234,
        8.690813064575195,
        11.072382926940918,
        10.169146537780762,
        12.156143188476562,
        13.450006484985352,
        14.443474769592285,
        21.0130672454834,
        22.124542236328125,
        26.07611083984375,
        31.450468063354492,
        47.11583709716797,
        43.25490188598633,
        29.676082611083984
      ],
      "student_per_layer": [
        0.5787713527679443,
        1.3789355754852295,
        1.902231216430664,
        1.9514288902282715,
        1.9535497426986694,
        2.0491881370544434,
        2.149158000946045,
        2.1966593265533447,
        2.1190524101257324,
        2.0256245136260986,
        2.2644553184509277,
        2.0249693393707275,
        2.194815158843994,
        2.785619020462036,
        3.470371723175049,
        3.83447265625
      ],
      "ratio_mean": 1.1160303354263306
    }
  },
  "method_means": {
    "ridge_kv_both": 0.3123431393866423,
    "ridge_self_kv": 0.32821237488919375,
    "student": 0.3282495887884593,
    "teacher": 0.807396388985802,
    "text": 0.3282495887884593
  },
  "ppl_means": {
    "ridge_kv_both": 29.0456752388901,
    "ridge_self_kv": 23.136744193421382,
    "student": 30.79558272604409
  },
  "chg": -0.015906449401816958,
  "ridge_kv_both_score_mean": 0.3123431393866423,
  "student_score_mean": 0.3282495887884593
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
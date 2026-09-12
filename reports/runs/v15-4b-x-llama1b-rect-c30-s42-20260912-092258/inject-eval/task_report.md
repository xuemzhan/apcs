# Task Report

- TASK_ID: `inject-eval`
- STATUS: **PASS**
- Generated at: 2026-09-12T09:28:13.010309

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
    "type": "rect_affine",
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
  "n_samples": 100,
  "n_capability_records": 500,
  "zero_prefill_verified": true,
  "protocol_version": "1.1",
  "git_hash": "3645d39e3b48905348908cecafa99a13fe34a764",
  "online": false,
  "calib_eval_disjoint": true,
  "rope_align": "rotated",
  "method_stats": {
    "ridge_kv_both": {
      "total": 100,
      "accuracy": 0.26,
      "gold_prob_mean": 0.25449736249168176,
      "conf_mean": 0.3222517949431708,
      "ppl_mean": 30.87055615683669,
      "logit_cos_mean": 0.9902417923233929
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
      "mean": 7.439561363990683e-05,
      "ci_low": -0.011620603738379327,
      "ci_high": 0.011934107052485904,
      "p": 0.5054945054945055,
      "n": 100,
      "tgrr": 0.00017051643677900118
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
      "mean": -0.04,
      "ci_low": -0.16,
      "ci_high": 0.09,
      "p": 0.7672327672327672,
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
    "chg": 7.439561363990683e-05,
    "ci_low": -0.011620603738379327,
    "ci_high": 0.011934107052485904,
    "p": 0.5054945054945055,
    "gate": "FAIL"
  },
  "psr": {
    "mean": -10.081048569241966,
    "median": -6.287691925667122,
    "n": 100,
    "note": "t_handoff 含测量专用 PPL forward（部署时不需要），PSR 偏保守；校准阶段 student self-prefill 成本未计入（见 mapper 校准成本）。"
  },
  "kv_norm_diagnostics": {
    "K": {
      "teacher_per_layer": [
        149.28839111328125,
        33.415096282958984,
        18.571510314941406,
        22.398962020874023,
        33.251258850097656,
        51.06267547607422,
        26.48597526550293,
        21.474153518676758,
        33.58911895751953,
        20.333837509155273,
        29.102876663208008,
        29.336687088012695,
        24.223548889160156,
        20.90026092529297,
        25.985498428344727,
        21.95773696899414,
        21.002248764038086,
        22.05470848083496,
        21.988277435302734,
        22.740142822265625,
        22.064613342285156,
        22.868274688720703,
        23.423032760620117,
        26.213815689086914,
        22.100061416625977,
        23.923583984375,
        20.40736198425293,
        19.779268264770508,
        20.628061294555664,
        21.098722457885742,
        22.949302673339844,
        21.30717658996582,
        20.920928955078125,
        20.20033836364746,
        21.29007911682129,
        31.118022918701172
      ],
      "student_per_layer": [
        14.696223258972168,
        18.60394859313965,
        17.33498191833496,
        18.953166961669922,
        18.56981658935547,
        19.858671188354492,
        19.908166885375977,
        16.691415786743164,
        18.564382553100586,
        18.769651412963867,
        17.351083755493164,
        20.06494140625,
        18.205373764038086,
        17.589019775390625,
        15.611200332641602,
        17.201229095458984
      ],
      "ratio_mean": 0.6652668714523315
    },
    "V": {
      "teacher_per_layer": [
        0.3492605984210968,
        0.5695530772209167,
        0.8092239499092102,
        1.0959709882736206,
        1.4074451923370361,
        1.5878186225891113,
        2.0199310779571533,
        3.0261943340301514,
        3.930877685546875,
        4.075098514556885,
        5.303684711456299,
        3.658644914627075,
        3.9954147338867188,
        3.9126782417297363,
        4.595703125,
        4.872912883758545,
        6.2928876876831055,
        5.587831020355225,
        6.285163402557373,
        6.4757399559021,
        6.271726131439209,
        7.072560787200928,
        8.743114471435547,
        8.932586669921875,
        11.408939361572266,
        10.482327461242676,
        12.546659469604492,
        13.877118110656738,
        14.8912935256958,
        21.710783004760742,
        22.833906173706055,
        26.935436248779297,
        32.48427200317383,
        48.68537902832031,
        44.632266998291016,
        30.586078643798828
      ],
      "student_per_layer": [
        0.5783166289329529,
        1.3787132501602173,
        1.90358304977417,
        1.95074462890625,
        1.9501011371612549,
        2.044100046157837,
        2.145930051803589,
        2.1914162635803223,
        2.113590955734253,
        2.022995710372925,
        2.2630679607391357,
        2.0279629230499268,
        2.1930882930755615,
        2.7901079654693604,
        3.470654010772705,
        3.8295202255249023
      ],
      "ratio_mean": 1.0929021835327148
    }
  },
  "method_means": {
    "ridge_kv_both": 0.3222517949431708,
    "ridge_self_kv": 0.32821237488919375,
    "student": 0.3282495887884593,
    "teacher": 0.807396388985802,
    "text": 0.3282495887884593
  },
  "ppl_means": {
    "ridge_kv_both": 30.87055615683669,
    "ridge_self_kv": 23.136744193421382,
    "student": 30.79558272604409
  },
  "chg": -0.005997793845288479,
  "ridge_kv_both_score_mean": 0.3222517949431708,
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
# Task Report

- TASK_ID: `t09`
- STATUS: **[SIMULATED] FAIL**
- Generated at: 2026-08-18T23:44:43.318321

> ⚠️ **本任务结果为离线模拟/占位数据（offline demo / placeholder），不可作为真实实验证据**（design.md §75）。

## OBJECTIVE
CHG / TGRR / JCR 主结果 + bootstrap CI + permutation test + Gap strata 报告（Gate 2A）。

## MODEL_PAIR
- Teacher: `Qwen/Qwen3-4B`
- Student: `Qwen/Qwen3-1.7B`

## DATASET
mmlu

## CONFIG
```json
{
  "context_lengths": [
    128
  ],
  "seeds": [
    0,
    1,
    2
  ],
  "mapper": {
    "type": "ridge",
    "rank": 4,
    "separate_kv": true,
    "de_rope": true,
    "source_top_k": 2,
    "layer_selection": "proportional",
    "alpha_max": 0.5,
    "rms_calibration": true,
    "shared_basis": false,
    "t06_aggregate_samples": 2,
    "calibration_samples": 8,
    "replacement_calib_samples": 8,
    "replacement_eval_samples": 4,
    "t04_eval_samples": 4,
    "t06_context": 128,
    "calibration_context": 128,
    "t06_ranks": [
      4,
      8
    ]
  },
  "advantage": {
    "rank": 2,
    "key": "lowrank",
    "value": "lowrank",
    "source_mixer": true,
    "rms_calibration": true,
    "bounded_alpha": true
  }
}
```

## IMPLEMENTATION
T09 Main Capability

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
  "task": "T09",
  "student_score": 0.5028901422325561,
  "teacher_score": 0.80689080056413,
  "teacher_gap": 0.30400065833157386,
  "per_method": [
    {
      "method": "text",
      "score": 0.5536508036380099,
      "score_std": 0.03214098142655722,
      "retention": 1.1009378731905626,
      "chg": 0.050760661405453766,
      "tgrr": 0.16697549829016836,
      "jcr_vs_student": 0.4166666666666667
    },
    {
      "method": "ridge",
      "score": 0.5898798996060576,
      "score_std": 0.02194620639341651,
      "retention": 1.172979643202618,
      "chg": 0.08698975737350145,
      "tgrr": 0.286149897999963,
      "jcr_vs_student": 0.5
    },
    {
      "method": "base_only",
      "score": 0.6056704085542352,
      "score_std": 0.03128753435881694,
      "retention": 1.2043791629428071,
      "chg": 0.10278026632167903,
      "tgrr": 0.33809224916077807,
      "jcr_vs_student": 0.5
    },
    {
      "method": "base_plus_adv",
      "score": 0.6727389721580916,
      "score_std": 0.022653634363264053,
      "retention": 1.337745395388941,
      "chg": 0.16984882992553552,
      "tgrr": 0.5587120464070877,
      "jcr_vs_student": 0.3333333333333333
    },
    {
      "method": "full_apcs",
      "score": 0.6767356129515711,
      "score_std": 0.023131687101710766,
      "retention": 1.3456927390686892,
      "chg": 0.17384547071901502,
      "tgrr": 0.5718588626522038,
      "jcr_vs_student": 0.6666666666666666
    }
  ],
  "chg_bootstrap": {
    "point": 0.16984882992553552,
    "ci_low": 0.14522589320549315,
    "ci_high": 0.2059797985734974,
    "ci": 0.95,
    "n_samples": 4
  },
  "chg_permutation": {
    "p_value": 0.057971014492753624,
    "n_perm": 2000
  },
  "seeds": [
    0,
    1,
    2
  ],
  "n_samples_per_seed": 4,
  "gap_strata": {
    "low": {
      "n": 1,
      "student_score": 0.5486487749415221,
      "teacher_score": 0.8344587493352607,
      "gap_mean": 0.28580997439373856,
      "base_plus_adv_chg": 0.15384639325383143,
      "base_plus_adv_tgrr": 0.5382821001267402
    },
    "medium": {
      "n": 1,
      "student_score": 0.48830295106347976,
      "teacher_score": 0.7784057346476031,
      "gap_mean": 0.29010278358412334,
      "base_plus_adv_chg": 0.16283520274140062,
      "base_plus_adv_tgrr": 0.5613017590856105
    },
    "high": {
      "n": 2,
      "student_score": 0.4873044214626113,
      "teacher_score": 0.8073493591368281,
      "gap_mean": 0.3200449376742168,
      "base_plus_adv_chg": 0.181356861853455,
      "base_plus_adv_tgrr": 0.5666606170101759
    }
  },
  "offline_demo": true,
  "note": "T09 得分为合成数据（offline demo），不可作为真实能力迁移的证据；design.md §75 要求真实 Test 上 CHG>0 才支持 Runtime Capability Transfer",
  "gate2a": "FAIL"
}
```

## STATISTICAL_CHECK
{
  "n_samples": 4,
  "seeds": [
    0,
    1,
    2
  ],
  "ci": 0.95
}

## BEHAVIOR_CHECK
{
  "jcr": "见 metrics.jcr_vs_student"
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
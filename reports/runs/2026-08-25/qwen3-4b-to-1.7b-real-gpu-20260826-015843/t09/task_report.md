# Task Report

- TASK_ID: `t09`
- STATUS: **[SIMULATED] PASS**
- Generated at: 2026-08-26T02:21:34.698489

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
    512,
    1024
  ],
  "seeds": [
    0,
    1,
    2
  ],
  "mapper": {
    "alpha_max": 0.5,
    "de_rope": true,
    "de_rope_k": true,
    "de_rope_v": false,
    "layer_selection": "proportional",
    "rank": 64,
    "real_calibration_samples": 64,
    "real_eval_samples": 16,
    "ridge_lambda_k": 0.001,
    "ridge_lambda_v": 0.001,
    "separate_kv": true,
    "shared_basis": true,
    "source_top_k": 2,
    "t06_aggregate_samples": 8,
    "type": "ridge"
  },
  "advantage": {
    "bounded_alpha": true,
    "key": "lowrank",
    "rank": 16,
    "rms_calibration": true,
    "source_mixer": true,
    "value": "lowrank"
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
- `provider.json`
- `stdout.log`
- `summary.md`

## KEY_METRICS
```json
{
  "task": "T09",
  "student_score": 0.4933231915660773,
  "teacher_score": 0.7953477591521788,
  "teacher_gap": 0.3020245675861015,
  "per_method": [
    {
      "method": "text",
      "score": 0.5505294844233126,
      "score_std": 0.026360692805553882,
      "retention": 1.1159610856234659,
      "chg": 0.05720629285723533,
      "tgrr": 0.18940940240209728,
      "jcr_vs_student": 0.5416666666666666
    },
    {
      "method": "ridge",
      "score": 0.5775613793139425,
      "score_std": 0.025012781732959614,
      "retention": 1.1707565936246522,
      "chg": 0.08423818774786518,
      "tgrr": 0.27891170715392366,
      "jcr_vs_student": 0.4375
    },
    {
      "method": "base_only",
      "score": 0.6111354005909186,
      "score_std": 0.02423425823136402,
      "retention": 1.2388134412469867,
      "chg": 0.11781220902484135,
      "tgrr": 0.3900749199525145,
      "jcr_vs_student": 0.4479166666666667
    },
    {
      "method": "base_plus_adv",
      "score": 0.6559393909952871,
      "score_std": 0.025216409022866856,
      "retention": 1.3296342077755905,
      "chg": 0.1626161994292098,
      "tgrr": 0.5384204362211394,
      "jcr_vs_student": 0.5
    },
    {
      "method": "full_apcs",
      "score": 0.7048409040687089,
      "score_std": 0.028675686595302056,
      "retention": 1.4287609342491256,
      "chg": 0.21151771250263163,
      "tgrr": 0.7003328046892474,
      "jcr_vs_student": 0.5729166666666666
    }
  ],
  "chg_bootstrap": {
    "point": 0.16261619942920977,
    "ci_low": 0.14926558262109155,
    "ci_high": 0.17692700218055152,
    "ci": 0.95,
    "n_samples": 32
  },
  "chg_permutation": {
    "p_value": 0.0004997501249375312,
    "n_perm": 2000
  },
  "seeds": [
    0,
    1,
    2
  ],
  "n_samples_per_seed": 32,
  "gap_strata": {
    "low": {
      "n": 11,
      "student_score": 0.5128866166076246,
      "teacher_score": 0.7722789487720777,
      "gap_mean": 0.2593923321644532,
      "base_plus_adv_chg": 0.14802476647511464,
      "base_plus_adv_tgrr": 0.5706597617591405
    },
    "medium": {
      "n": 10,
      "student_score": 0.49112943610013976,
      "teacher_score": 0.7957881984009313,
      "gap_mean": 0.30465876230079153,
      "base_plus_adv_chg": 0.15113482511335746,
      "base_plus_adv_tgrr": 0.49607903600731196
    },
    "high": {
      "n": 11,
      "student_score": 0.4757540896753824,
      "teacher_score": 0.8180161702152321,
      "gap_mean": 0.34226208053984963,
      "base_plus_adv_chg": 0.18764524539771593,
      "base_plus_adv_tgrr": 0.5482501745496996
    }
  },
  "offline_demo": true,
  "note": "T09 得分为合成数据（offline demo），不可作为真实能力迁移的证据；design.md §75 要求真实 Test 上 CHG>0 才支持 Runtime Capability Transfer",
  "gate2a": "PASS"
}
```

## STATISTICAL_CHECK
{
  "n_samples": 32,
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
  "gate": "PASS"
}

## FAILURE_ANALYSIS
(无)

## NEXT_ALLOWED_TASK
t10
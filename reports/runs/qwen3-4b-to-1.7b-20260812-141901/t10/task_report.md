# Task Report

- TASK_ID: `t10`
- STATUS: **[SIMULATED] OK**
- Generated at: 2026-08-12T14:45:00.345921

> ⚠️ **本任务结果为离线模拟/占位数据（offline demo / placeholder），不可作为真实实验证据**（design.md §75）。

## OBJECTIVE
Scenario A/B/C + PSR_A + N_BE + cache_bytes/VRAM/RAM + §53 单卡流程。

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
T10 System Cost

## OUTPUT_FILES
- `compliance.json`
- `config.json`
- `metadata.json`
- `metrics.json`
- `stdout.log`
- `summary.md`
- `system.json`
- `task_report.md`

## KEY_METRICS
```json
{
  "task": "T10",
  "contexts": [
    128
  ],
  "repeats": 2,
  "warmup": 1,
  "per_context": [
    {
      "context": 128,
      "psr_a_p50": 0.6033970276008491,
      "psr_a_p95": 0.6033970276008491,
      "n_be_median": 4,
      "cost_b_p50": 11.0304,
      "cost_b_p95": 11.0304,
      "teacher_cache_bytes": 18874368,
      "student_cache_bytes": 14680064
    }
  ],
  "vram_teacher_mb_est": 0.28125,
  "ram_student_mb_est": 0.21875,
  "scenario_definitions": {
    "A": "natural handoff; PSR_A = 1 - (T_map+T_load+T_query)/T_prefill_S",
    "B": "teacher-for-transfer; Cost = T_prefill_T + T_map + T_load + T_query",
    "C": "one-teacher-many-students; N_BE = ceil((T_prefill_T + T_map) / (T_prefill_S - T_load - T_query))"
  },
  "single_card_pipeline_order": [
    "teacher_load",
    "forward",
    "capture",
    "cpu_offload",
    "teacher_unload",
    "cuda_cleanup",
    "student_load",
    "map",
    "inject",
    "decode"
  ],
  "offline_demo": true,
  "note": "T10 全部耗时由 _simulate_timings 线性公式生成，VRAM/RAM 为量级估算，PSR_A / Cost_B / N_BE 均为推导值而非实测；design.md §38/§75 禁止将理论估算作为论文系统收益结果。"
}
```

## STATISTICAL_CHECK
{
  "n_samples": null,
  "seeds": [
    0,
    1,
    2
  ],
  "ci": 0.95
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
{
  "task": "T10",
  "contexts": [
    128
  ],
  "repeats": 2,
  "warmup": 1,
  "per_context": [
    {
      "context": 128,
      "psr_a_p50": 0.6033970276008491,
      "psr_a_p95": 0.6033970276008491,
      "n_be_median": 4,
      "cost_b_p50": 11.0304,
      "cost_b_p95": 11.0304,
      "teacher_cache_bytes": 18874368,
      "student_cache_bytes": 14680064
    }
  ],
  "vram_teacher_mb_est": 0.28125,
  "ram_student_mb_est": 0.21875,
  "scenario_definitions": {
    "A": "natural handoff; PSR_A = 1 - (T_map+T_load+T_query)/T_prefill_S",
    "B": "teacher-for-transfer; Cost = T_prefill_T + T_map + T_load + T_query",
    "C": "one-teacher-many-students; N_BE = ceil((T_prefill_T + T_map) / (T_prefill_S - T_load - T_query))"
  },
  "single_card_pipeline_order": [
    "teacher_load",
    "forward",
    "capture",
    "cpu_offload",
    "teacher_unload",
    "cuda_cleanup",
    "student_load",
    "map",
    "inject",
    "decode"
  ],
  "offline_demo": true,
  "note": "T10 全部耗时由 _simulate_timings 线性公式生成，VRAM/RAM 为量级估算，PSR_A / Cost_B / N_BE 均为推导值而非实测；design.md §38/§75 禁止将理论估算作为论文系统收益结果。"
}

## ACCEPTANCE_CRITERIA
{
  "gate": "OK"
}

## FAILURE_ANALYSIS
(无)

## NEXT_ALLOWED_TASK
t12
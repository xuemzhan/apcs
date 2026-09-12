# Task Report

- TASK_ID: `t10`
- STATUS: **[SIMULATED] OK**
- Generated at: 2026-08-26T02:21:55.638063

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
T10 System Cost

## OUTPUT_FILES
- `compliance.json`
- `config.json`
- `metadata.json`
- `metrics.json`
- `provider.json`
- `stdout.log`
- `summary.md`
- `system.json`

## KEY_METRICS
```json
{
  "task": "T10",
  "contexts": [
    1024,
    4096,
    8192,
    16384
  ],
  "repeats": 10,
  "warmup": 2,
  "per_context": [
    {
      "context": 1024,
      "psr_a_p50": 0.6166666666666667,
      "psr_a_p95": 0.6166666666666667,
      "n_be_median": 3,
      "cost_b_p50": 14.6432,
      "cost_b_p95": 14.6432,
      "teacher_cache_bytes": 150994944,
      "student_cache_bytes": 117440512
    },
    {
      "context": 4096,
      "psr_a_p50": 0.6166666666666667,
      "psr_a_p95": 0.6166666666666667,
      "n_be_median": 3,
      "cost_b_p50": 58.5728,
      "cost_b_p95": 58.5728,
      "teacher_cache_bytes": 603979776,
      "student_cache_bytes": 469762048
    },
    {
      "context": 8192,
      "psr_a_p50": 0.6166666666666667,
      "psr_a_p95": 0.6166666666666667,
      "n_be_median": 3,
      "cost_b_p50": 117.1456,
      "cost_b_p95": 117.1456,
      "teacher_cache_bytes": 1207959552,
      "student_cache_bytes": 939524096
    },
    {
      "context": 16384,
      "psr_a_p50": 0.6166666666666667,
      "psr_a_p95": 0.6166666666666667,
      "n_be_median": 3,
      "cost_b_p50": 234.2912,
      "cost_b_p95": 234.2912,
      "teacher_cache_bytes": 2415919104,
      "student_cache_bytes": 1879048192
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
  "timing_evidence": "cuda_proxy_not_end_to_end",
  "note": "T10 使用 CUDA 代理算子诊断，不是 HandoffPipeline 端到端计时；PSR_A / Cost_B / N_BE 仍不可作为真实系统收益证据。"
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
    1024,
    4096,
    8192,
    16384
  ],
  "repeats": 10,
  "warmup": 2,
  "per_context": [
    {
      "context": 1024,
      "psr_a_p50": 0.6166666666666667,
      "psr_a_p95": 0.6166666666666667,
      "n_be_median": 3,
      "cost_b_p50": 14.6432,
      "cost_b_p95": 14.6432,
      "teacher_cache_bytes": 150994944,
      "student_cache_bytes": 117440512
    },
    {
      "context": 4096,
      "psr_a_p50": 0.6166666666666667,
      "psr_a_p95": 0.6166666666666667,
      "n_be_median": 3,
      "cost_b_p50": 58.5728,
      "cost_b_p95": 58.5728,
      "teacher_cache_bytes": 603979776,
      "student_cache_bytes": 469762048
    },
    {
      "context": 8192,
      "psr_a_p50": 0.6166666666666667,
      "psr_a_p95": 0.6166666666666667,
      "n_be_median": 3,
      "cost_b_p50": 117.1456,
      "cost_b_p95": 117.1456,
      "teacher_cache_bytes": 1207959552,
      "student_cache_bytes": 939524096
    },
    {
      "context": 16384,
      "psr_a_p50": 0.6166666666666667,
      "psr_a_p95": 0.6166666666666667,
      "n_be_median": 3,
      "cost_b_p50": 234.2912,
      "cost_b_p95": 234.2912,
      "teacher_cache_bytes": 2415919104,
      "student_cache_bytes": 1879048192
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
  "timing_evidence": "cuda_proxy_not_end_to_end",
  "note": "T10 使用 CUDA 代理算子诊断，不是 HandoffPipeline 端到端计时；PSR_A / Cost_B / N_BE 仍不可作为真实系统收益证据。"
}

## ACCEPTANCE_CRITERIA
{
  "gate": "OK"
}

## FAILURE_ANALYSIS
(无)

## NEXT_ALLOWED_TASK
t12
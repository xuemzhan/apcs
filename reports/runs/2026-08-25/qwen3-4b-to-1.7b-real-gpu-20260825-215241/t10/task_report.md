# Task Report

- TASK_ID: `t10`
- STATUS: **[SIMULATED] OK**
- Generated at: 2026-08-25T23:10:34.997225

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
    "type": "ridge",
    "rank": 32,
    "separate_kv": true,
    "de_rope": true,
    "de_rope_k": true,
    "de_rope_v": false,
    "ridge_lambda_k": 0.001,
    "ridge_lambda_v": 0.01,
    "real_calibration_samples": 32,
    "real_eval_samples": 16,
    "source_top_k": 2,
    "layer_selection": "proportional",
    "alpha_max": 0.5,
    "rms_calibration": true,
    "shared_basis": false,
    "t06_aggregate_samples": 8
  },
  "advantage": {
    "rank": 16,
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
- `provider.json`
- `stdout.log`
- `summary.md`
- `system.json`

## KEY_METRICS
```json
{
  "task": "T10",
  "contexts": [
    8192,
    16384
  ],
  "repeats": 10,
  "warmup": 2,
  "per_context": [
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
    8192,
    16384
  ],
  "repeats": 10,
  "warmup": 2,
  "per_context": [
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
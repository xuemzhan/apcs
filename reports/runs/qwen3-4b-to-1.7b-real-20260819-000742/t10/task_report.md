# Task Report

- TASK_ID: `t10`
- STATUS: **OK**
- Generated at: 2026-08-19T00:23:36.093937

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
    "rank": 16,
    "separate_kv": true,
    "de_rope": true,
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
  "vram_teacher_mb_est": 24.0,
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
  "offline_demo": false,
  "note": "T10 为真实 CUDA 计时（§49 warmup + ≥10 repeats + P50/P95），VRAM 为 torch.cuda.max_memory_allocated 实测。"
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
  "vram_teacher_mb_est": 24.0,
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
  "offline_demo": false,
  "note": "T10 为真实 CUDA 计时（§49 warmup + ≥10 repeats + P50/P95），VRAM 为 torch.cuda.max_memory_allocated 实测。"
}

## ACCEPTANCE_CRITERIA
{
  "gate": "OK"
}

## FAILURE_ANALYSIS
(无)

## NEXT_ALLOWED_TASK
t12
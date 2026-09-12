#!/bin/bash
# 优化实验脚本：系统性测试不同配置对 V retention 的影响
# 每轮只改一个变量，记录结果

CONFIG_BASE="configs/pair_qwen3_real_gpu.yaml"
RUN_DIR="reports/runs/2026-08-25"

# 创建优化结果记录文件
LOG="$RUN_DIR/optimization_log.md"
cat > "$LOG" << 'EOF'
# V Retention 优化实验日志

## 基线 (2026-08-25)
- retention_K=0.965, retention_V=0.563, mean=0.764
- Gate 1: FAIL (need >= 0.90)

## 实验矩阵
| 轮次 | 变量 | retention_K | retention_V | mean_retention | Gate 1 |
|------|------|-------------|-------------|----------------|--------|
EOF

run_t04_t05() {
    local label="$1"
    local run_id=$(cat "$RUN_DIR/qwen3-4b-to-1.7b-real-gpu.current" 2>/dev/null)
    
    echo "=== $label ==="
    echo "run_id: $run_id"
    
    # Run T04
    python -m apcs.cli t04 --config "$CONFIG_BASE" 2>&1 | grep -E "retention_K|retention_V|mean_retention|gate" | head -5
    
    # Run T05
    python -m apcs.cli t05 --config "$CONFIG_BASE" 2>&1 | grep -E "retention_K|retention_V|mean_retention|gate1" | head -5
    echo "---"
}

# 轮次1: rank=64, calib=64
echo "=== 轮次1: rank=64, calib=64 ==="
cat > "$CONFIG_BASE" << 'YAMLEOF'
experiment:
  name: qwen3-4b-to-1.7b-real-gpu
  run_id: ${experiment.name}-${run.timestamp}
  description: |
    V retention 优化实验 - 轮次1: rank=64, calib=64

provider:
  kv: hf
  score: synthetic
  timing: hf

teacher:
  model_id: Qwen/Qwen3-4B
  revision: main
  dtype: bfloat16
  attention_implementation: sdpa
  device_map: cuda:0
  num_layers: 36
  num_kv_heads: 8
  head_dim: 128

student:
  model_id: Qwen/Qwen3-1.7B
  revision: main
  dtype: bfloat16
  attention_implementation: sdpa
  device_map: cuda:0
  freeze: true
  num_layers: 28
  num_kv_heads: 8
  head_dim: 128

datasets:
  fidelity:
    - hellaswag
    - arc_challenge
    - winogrande
  teacher_advantage:
    primary: mmlu
    splits: [train, validation, test]

context_lengths: [512, 1024]

mapper:
  type: ridge
  rank: 64
  separate_kv: true
  de_rope: true
  de_rope_k: true
  de_rope_v: false
  ridge_lambda_k: 0.001
  ridge_lambda_v: 0.001
  real_calibration_samples: 64
  real_eval_samples: 16
  source_top_k: 2
  layer_selection: proportional
  alpha_max: 0.5
  rms_calibration: true
  shared_basis: false
  t06_aggregate_samples: 8

dataset_prepare:
  samples_per_split: 128
  seed: 0

advantage:
  rank: 16
  key: lowrank
  value: lowrank
  source_mixer: true
  rms_calibration: true
  bounded_alpha: true

query_gate:
  enabled: false

loss:
  lambda_task: 1.0
  lambda_self: 0.1
  lambda_teacher: 0.1
  lambda_att: 1.0
  lambda_reg: 0.01
  teacher_weight_from: train_split_margin

seeds: [0, 1, 2]
statistics:
  bootstrap_n: 1000
  ci: 0.95
  paired_test: permutation

timing:
  warmup: 2
  repeats: 10
  sync_cuda: true
  report: [p50, p95]

cache_residency: cpu
kv_save_full_debug:
  max_samples: 10

output:
  base_dir: reports/runs/2026-08-25
  files:
    - config.yaml
    - metrics.json
    - system.json
    - geometry.json
    - stdout.log
    - summary.md

gates:
  retention_min: 0.90
  retention_strong: 0.95
  chg_positive: true
YAMLEOF

python -m apcs.cli t00 --config "$CONFIG_BASE" --new-run 2>&1 | tail -2
python -m apcs.cli t01 --config "$CONFIG_BASE" 2>&1 | tail -2
python -m apcs.cli t02 --config "$CONFIG_BASE" 2>&1 | tail -2
python -m apcs.cli t03 --config "$CONFIG_BASE" 2>&1 | tail -2

echo "--- T04/T05 rank=64, calib=64 ---"
python -m apcs.cli t04 --config "$CONFIG_BASE" 2>&1 | tail -15
python -m apcs.cli t05 --config "$CONFIG_BASE" 2>&1 | tail -15

#!/bin/bash
set -e

CONFIG="configs/pair_qwen3_real_gpu.yaml"
RUN_ID="qwen3-4b-to-1.7b-real-gpu-20260825-215241"

echo "=== Running all tasks with run_id: $RUN_ID ==="

# T00 - already done
echo "--- T00 ---"
python -m apcs.cli t00 --config $CONFIG --run-id $RUN_ID || true

# T01 - Self-KV Replay (Gate 0)
echo "--- T01 ---"
python -m apcs.cli t01 --config $CONFIG --run-id $RUN_ID

# T02 - RoPE Round-trip
echo "--- T02 ---"
python -m apcs.cli t02 --config $CONFIG --run-id $RUN_ID

# T03 - Layer Alignment
echo "--- T03 ---"
python -m apcs.cli t03 --config $CONFIG --run-id $RUN_ID

# T04 - Ridge Baseline
echo "--- T04 ---"
python -m apcs.cli t04 --config $CONFIG --run-id $RUN_ID

# T05 - Replacement (Gate 1)
echo "--- T05 ---"
python -m apcs.cli t05 --config $CONFIG --run-id $RUN_ID

# T06 - Lightweight Mapper
echo "--- T06 ---"
python -m apcs.cli t06 --config $CONFIG --run-id $RUN_ID

# T07 - Teacher Gap Freeze
echo "--- T07 ---"
python -m apcs.cli t07 --config $CONFIG --run-id $RUN_ID

# T08 - Advantage State Training
echo "--- T08 ---"
python -m apcs.cli t08 --config $CONFIG --run-id $RUN_ID

# T09 - Main Capability (Gate 2A)
echo "--- T09 ---"
python -m apcs.cli t09 --config $CONFIG --run-id $RUN_ID

# T10 - System Cost
echo "--- T10 ---"
python -m apcs.cli t10 --config $CONFIG --run-id $RUN_ID

# T11 - MVP Decision
echo "--- T11 ---"
python -m apcs.cli t11 --config $CONFIG --run-id $RUN_ID

# T12 - Geometry Diagnostics
echo "--- T12 ---"
python -m apcs.cli t12 --config $CONFIG --run-id $RUN_ID

# T13 - Generalization
echo "--- T13 ---"
python -m apcs.cli t13 --config $CONFIG --run-id $RUN_ID

echo "=== All tasks completed ==="

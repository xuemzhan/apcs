#!/usr/bin/env bash
# Run cross-model KV handoff experiments using existing apcs CLI infrastructure.
# All 4 Qwen3 model pairs, real paired calibration, Ridge per-head mapper.
set -euo pipefail

cd /workspace/apcs

CONFIGS=(
    "configs/crossmodel_4b_1.7b.yaml"
    "configs/crossmodel_4b_0.6b.yaml"
    "configs/crossmodel_8b_1.7b.yaml"
    "configs/crossmodel_8b_0.6b.yaml"
)

for cfg in "${CONFIGS[@]}"; do
    echo ""
    echo "================================================================"
    echo "Running: $cfg"
    echo "================================================================"
    python -m apcs.cli inject-eval \
        --config "$cfg" \
        --new-run \
        --force \
        2>&1 | tee "/tmp/apcs_$(basename "$cfg" .yaml)_$(date +%Y%m%d_%H%M%S).log"
    echo ""
    echo "Done: $cfg"
done

echo ""
echo "================================================================"
echo "All experiments completed!"
echo "================================================================"

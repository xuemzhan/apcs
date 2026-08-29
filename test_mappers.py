#!/usr/bin/env python3
"""Test different mapper architectures on 4B→1.7B pair."""
import subprocess
import sys

# Configs for different mapper types
configs = {
    "ridge_perhead": {"type": "ridge"},
    "ridge_layer": {"type": "ridge_layer"},
    "lowrank_r8": {"type": "lowrank", "rank": 8},
    "lowrank_r16": {"type": "lowrank", "rank": 16},
    "lowrank_r32": {"type": "lowrank", "rank": 32},
    "affine": {"type": "affine"},
}

import yaml

for name, mapper_cfg in configs.items():
    cfg = {
        "experiment": {
            "name": f"mapper-{name}",
            "run_id": "${experiment.name}-${run.timestamp}",
        },
        "provider": {"kv": "hf", "score": "synthetic", "timing": "hf"},
        "teacher": {
            "model_id": "Qwen/Qwen3-4B",
            "revision": "main",
            "dtype": "float16",
            "attention_implementation": "sdpa",
            "device_map": "cuda:0",
            "num_layers": 36,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
        "student": {
            "model_id": "Qwen/Qwen3-1.7B",
            "revision": "main",
            "dtype": "float16",
            "attention_implementation": "sdpa",
            "device_map": "cuda:0",
            "freeze": True,
            "num_layers": 28,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
        "datasets": {
            "fidelity": ["hellaswag", "arc_challenge", "winogrande"],
            "teacher_advantage": {"primary": "mmlu"},
        },
        "context_lengths": [512],
        "mapper": {
            **mapper_cfg,
            "separate_kv": True,
            "de_rope": True,
            "de_rope_k": True,
            "de_rope_v": False,
            "ridge_lambda_k": 0.001,
            "ridge_lambda_v": 0.001,
            "inject_eval_calib_samples": 20,
            "layer_selection": "proportional",
            "inject_eval": True,
        },
        "inject_eval": {
            "max_samples": 20,
            "seed": 0,
            "ablation_modes": ["kv_both"],
        },
        "seeds": [0],
        "output": {"base_dir": "reports/runs"},
    }
    
    cfg_path = f"configs/mapper_test_{name}.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    result = subprocess.run(
        [sys.executable, "-m", "apcs.cli", "inject-eval",
         "--config", cfg_path, "--new-run", "--force"],
        capture_output=True, text=True, timeout=600
    )
    
    # Extract key metrics
    for line in result.stdout.split("\n"):
        if '"chg"' in line or '"ridge_kv_both_score_mean"' in line or '"student_score_mean"' in line:
            print(line.strip())
        if "DONE" in line:
            print(line.strip())
        if "TypeError" in line or "Error" in line:
            print(f"  ERROR: {line.strip()}")
    
    if result.returncode != 0:
        # Print last few lines of stderr
        stderr_lines = result.stderr.strip().split("\n")
        for line in stderr_lines[-5:]:
            print(f"  STDERR: {line}")

print("\n\nAll mapper tests completed!")

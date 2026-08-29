#!/usr/bin/env python3
"""Generate multi-seed configs and run experiments."""
import subprocess
import sys

PAIRS = [
    ("4b", "1.7b", 36, 28),
    ("4b", "0.6b", 36, 28),
    ("8b", "1.7b", 36, 28),
    ("8b", "0.6b", 36, 28),
]

SEEDS = [42, 123, 456]
LAMBDAS = [1e-4, 1e-3, 1e-2]

results = []

for t_size, s_size, t_layers, s_layers in PAIRS:
    for seed in SEEDS:
        for lam in LAMBDAS:
            cfg_name = f"configs/sweep_{t_size}_{s_size}_s{seed}_l{lam:.0e}.yaml"
            t_name = f"Qwen3-{t_size.upper()}" if t_size != "0.6b" else "Qwen3-0.6B"
            s_name = f"Qwen3-{s_size.upper()}" if s_size != "0.6b" else "Qwen3-0.6B"
            if t_size == "4b":
                t_model = "Qwen/Qwen3-4B"
            elif t_size == "8b":
                t_model = "Qwen/Qwen3-8B"
            else:
                t_model = f"Qwen/Qwen3-{t_size}"
            if s_size == "0.6b":
                s_model = "Qwen/Qwen3-0.6B"
            else:
                s_model = f"Qwen/Qwen3-{s_size}"
            
            with open(cfg_name, "w") as f:
                f.write(f"""experiment:
  name: {t_size}-to-{s_size}-s{seed}
  run_id: ${{experiment.name}}-${{run.timestamp}}

provider:
  kv: hf
  score: synthetic
  timing: hf

teacher:
  model_id: {t_model}
  revision: main
  dtype: float16
  attention_implementation: sdpa
  device_map: cuda:0
  num_layers: {t_layers}
  num_kv_heads: 8
  head_dim: 128

student:
  model_id: {s_model}
  revision: main
  dtype: float16
  attention_implementation: sdpa
  device_map: cuda:0
  freeze: true
  num_layers: {s_layers}
  num_kv_heads: 8
  head_dim: 128

datasets:
  fidelity:
    - hellaswag
    - arc_challenge
    - winogrande
  teacher_advantage:
    primary: mmlu

context_lengths: [512]

mapper:
  type: ridge
  separate_kv: true
  de_rope: true
  de_rope_k: true
  de_rope_v: false
  ridge_lambda_k: {lam}
  ridge_lambda_v: {lam}
  inject_eval_calib_samples: 20
  layer_selection: proportional
  inject_eval: true

inject_eval:
  max_samples: 20
  seed: {seed}
  ablation_modes: [kv_both]

seeds: [{seed}]
output:
  base_dir: reports/runs
""")
            print(f"Generated: {cfg_name}")

# Now run all configs
print("\nRunning all sweep configs...")
for t_size, s_size, _, _ in PAIRS:
    for seed in SEEDS:
        for lam in LAMBDAS:
            cfg_name = f"configs/sweep_{t_size}_{s_size}_s{seed}_l{lam:.0e}.yaml"
            print(f"\n{'='*60}")
            print(f"Running: {cfg_name}")
            print(f"{'='*60}")
            result = subprocess.run(
                [sys.executable, "-m", "apcs.cli", "inject-eval",
                 "--config", cfg_name, "--new-run", "--force"],
                capture_output=True, text=True, timeout=600
            )
            # Extract metrics from output
            for line in result.stdout.split("\n"):
                if '"chg"' in line or '"ridge_kv_both_score_mean"' in line or '"student_score_mean"' in line:
                    print(line.strip())
                if "DONE" in line:
                    print(line.strip())
            if result.returncode != 0:
                print(f"ERROR: {result.stderr[-500:]}")
            # Save log
            log_name = f"/tmp/sweep_{t_size}_{s_size}_s{seed}_l{lam:.0e}.log"
            with open(log_name, "w") as f:
                f.write(result.stdout)
                f.write("\n--- STDERR ---\n")
                f.write(result.stderr)

print("\n\nAll sweep experiments completed!")

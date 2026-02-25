#!/usr/bin/env python3
"""
Astex 5-Target Quick Ablation Runner
Week 1-2 Sprint: no_backbone vs fksmc vs socm
Targets chosen from Astex Diverse Set (high-quality crystal structures)
"""
import subprocess
import json
import os
from pathlib import Path

TARGETS = ["4ecy", "7msa", "5m1l", "6y0t", "7qxr"]
STEPS = 300
CONFIGS = {
    "no_backbone": ["--steps", str(STEPS), "--no_backbone"],
    "fksmc":       ["--steps", str(STEPS), "--fksmc"],
    "socm":        ["--steps", str(STEPS), "--socm"],
    "fksmc_socm":  ["--steps", str(STEPS), "--fksmc", "--socm"],
}

results_all = {}

for cfg_name, extra_args in CONFIGS.items():
    print(f"\n{'='*50}")
    print(f"Running config: {cfg_name}")
    print(f"{'='*50}")
    
    targets_str = ",".join(TARGETS)
    cmd = [
        "python", "run_dockgen.py",
        "--targets", targets_str,
        *extra_args,
    ]
    print(f"CMD: {' '.join(cmd)}")
    ret = subprocess.run(cmd, capture_output=False, text=True)
    
    # Parse result
    # The output prefix varies by config
    if "--no_backbone" in extra_args:
        result_file = "dockgen_no_backbone_results.json"
    elif "--fksmc" in extra_args and "--socm" in extra_args:
        result_file = "dockgen_fksmc_results.json"
    elif "--fksmc" in extra_args:
        result_file = "dockgen_fksmc_results.json"
    elif "--socm" in extra_args:
        result_file = "dockgen_results.json"
    else:
        result_file = "dockgen_results.json"
    
    if os.path.exists(result_file):
        with open(result_file) as f:
            d = json.load(f)
        results_all[cfg_name] = d
        s = d.get("summary", {})
        print(f"  SR@2A={s.get('SR@2A',0):.1f}%  SR@5A={s.get('SR@5A',0):.1f}%  MedianRMSD={s.get('median_rmsd',99):.3f}A")

# Save consolidated
Path("./ablation_astex5_summary.json").write_text(
    json.dumps({"configs": list(CONFIGS.keys()), "results": results_all}, indent=2)
)
print("\n\nAblation Summary saved to ablation_astex5_summary.json")

# Print table
print("\n| Config | SR@2A | SR@5A | Median RMSD |")
print("|--------|-------|-------|-------------|")
for cfg, r in results_all.items():
    s = r.get("summary", {})
    print(f"| {cfg:14s} | {s.get('SR@2A', 0):5.1f}% | {s.get('SR@5A', 0):5.1f}% | {s.get('median_rmsd', 99):.3f}A |")

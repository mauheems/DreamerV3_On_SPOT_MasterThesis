#!/usr/bin/env python3
"""Debug checkpoint structure"""

import pickle
from pathlib import Path

ckpt_path = Path("/home/maurits-heemskerk/Documents/Uni/Master_Thesis/dreamer_results_local/ablation-all-combined-20260403-171108/checkpoint.ckpt")

with open(ckpt_path, 'rb') as f:
    checkpoint = pickle.load(f)

def explore_dict(obj, depth=0, max_depth=3, prefix=""):
    """Recursively explore checkpoint structure"""
    if depth > max_depth:
        return
    
    if isinstance(obj, dict):
        for k, v in list(obj.items())[:5]:  # Show first 5 keys
            full_path = f"{prefix}/{k}" if prefix else k
            if isinstance(v, dict):
                print("  " * depth + f"📁 {k}:")
                explore_dict(v, depth + 1, max_depth, full_path)
            elif isinstance(v, (list, tuple)):
                print("  " * depth + f"📋 {k}: {type(v).__name__}[{len(v)}]")
            else:
                print("  " * depth + f"📄 {k}: {type(v).__name__}")

print("Top-level structure:")
explore_dict(checkpoint, max_depth=2)

print("\n\nTop-level keys:")
print(list(checkpoint.keys()))

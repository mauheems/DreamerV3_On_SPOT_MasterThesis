#!/usr/bin/env python3
"""
Batch convert all DreamerV3 checkpoints in dreamer_results_local to float16 inference format.
"""

import sys
from pathlib import Path
import subprocess

def main():
    results_dir = Path("/home/maurits-heemskerk/Documents/Uni/Master_Thesis/dreamer_results_local")
    
    # Find all checkpoints
    checkpoints = sorted(results_dir.glob("**/checkpoint.ckpt"))
    
    if not checkpoints:
        print("❌ No checkpoints found!")
        sys.exit(1)
    
    print(f"🔍 Found {len(checkpoints)} checkpoints to convert\n")
    
    script_path = results_dir.parent / "convert_checkpoint_to_float16.py"
    
    for i, ckpt_path in enumerate(checkpoints, 1):
        output_path = ckpt_path.parent / "checkpoint_float16_inference.ckpt"
        
        # Skip if already converted
        if output_path.exists():
            print(f"[{i}/{len(checkpoints)}] ⏭️  {ckpt_path.parent.name}")
            print(f"    → Already converted (skipping)\n")
            continue
        
        print(f"[{i}/{len(checkpoints)}] 🔄 {ckpt_path.parent.name}")
        print(f"    Input:  {ckpt_path.name}")
        print(f"    Output: {output_path.name}")
        
        # Run conversion with both filtering and float16
        result = subprocess.run(
            [sys.executable, str(script_path), str(ckpt_path), str(output_path), "--inference"],
            capture_output=False,
            text=True
        )
        
        if result.returncode != 0:
            print(f"    ❌ Failed!\n")
            continue
        
        print()
    
    print("✅ Batch conversion complete!")


if __name__ == '__main__':
    main()

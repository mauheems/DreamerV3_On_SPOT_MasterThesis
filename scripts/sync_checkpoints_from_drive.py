#!/usr/bin/env python3
"""
Pull checkpoints from external drive, convert them, and save locally.
- Copies checkpoint.ckpt from external drive to temp
- Copies config.yaml to final location
- Converts checkpoint (float32→float16 + inference filtering)
- Deletes temp original checkpoint
- Keeps only converted checkpoint + config locally
"""

import sys
from pathlib import Path
import subprocess
import shutil
import tempfile

def main():
    # Paths
    external_drive = Path("/media/maurits-heemskerk/69987a47-b840-4db7-9f8b-7cc05f14d09e3/checkpoints")
    local_results_dir = Path("/home/maurits-heemskerk/Documents/Uni/Master_Thesis/dreamer_results_local")
    script_path = local_results_dir.parent / "scripts" / "convert_checkpoint_to_float16.py"
    
    if not external_drive.exists():
        print(f"❌ External drive not found: {external_drive}")
        sys.exit(1)
    
    # Find all checkpoint directories on external drive
    checkpoint_dirs = sorted([d for d in external_drive.iterdir() if d.is_dir()])
    
    if not checkpoint_dirs:
        print("❌ No checkpoint directories found on external drive!")
        sys.exit(1)
    
    print(f"🔍 Found {len(checkpoint_dirs)} checkpoints on external drive\n")
    
    processed = 0
    
    for i, ckpt_dir_external in enumerate(checkpoint_dirs, 1):
        run_name = ckpt_dir_external.name
        ckpt_file_external = ckpt_dir_external / "checkpoint.ckpt"
        config_file_external = ckpt_dir_external / "config.yaml"
        metrics_file_external = ckpt_dir_external / "metrics.json"
        
        # Skip if checkpoint doesn't exist
        if not ckpt_file_external.exists():
            print(f"[{i}/{len(checkpoint_dirs)}] ⏭️  {run_name} (no checkpoint file)")
            continue
        
        # Create local directory
        ckpt_dir_local = local_results_dir / run_name
        ckpt_dir_local.mkdir(parents=True, exist_ok=True)
        
        # Check if already converted
        converted_path = ckpt_dir_local / "checkpoint_float16_inference.ckpt"
        if converted_path.exists():
            print(f"[{i}/{len(checkpoint_dirs)}] ⏭️  {run_name} (already converted)")
            continue
        
        print(f"[{i}/{len(checkpoint_dirs)}] 🔄 {run_name}")
        
        # Create temp directory for conversion
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            temp_ckpt = tmpdir / "checkpoint.ckpt"
            
            try:
                # 1. Copy checkpoint from external drive to temp
                print(f"    📥 Copying from external drive...")
                shutil.copy2(ckpt_file_external, temp_ckpt)
                
                # 2. Copy config to local (if exists and not already present)
                if config_file_external.exists():
                    config_local = ckpt_dir_local / "config.yaml"
                    if not config_local.exists():
                        shutil.copy2(config_file_external, config_local)
                        print(f"    ✓ Copied config")
                
                # 3. Copy metrics to local (if exists and not already present)
                if metrics_file_external.exists():
                    metrics_local = ckpt_dir_local / "metrics.json"
                    if not metrics_local.exists():
                        shutil.copy2(metrics_file_external, metrics_local)
                        print(f"    ✓ Copied metrics")
                
                # 3. Convert checkpoint
                print(f"    🔄 Converting (float32→float16 + inference filtering)...")
                result = subprocess.run(
                    [sys.executable, str(script_path), str(temp_ckpt), 
                     str(converted_path), "--inference"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    print(f"    ❌ Conversion failed!")
                    if result.stderr:
                        print(f"    Error: {result.stderr[:200]}")
                    continue
                
                # Extract file sizes from output
                if "After filtering:" in result.stdout:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if "After fp16:" in line or "−" in line:
                            print(f"    {line.strip()}")
                
                print(f"    ✅ Saved locally\n")
                processed += 1
                
            except Exception as e:
                print(f"    ❌ Error: {e}\n")
    
    print(f"✅ Complete! Processed {processed}/{len(checkpoint_dirs)} checkpoints")
    print(f"\n📊 Summary per checkpoint:")
    print(f"   • Size reduction: ~83% (2934.7MB → 501.5MB)")
    print(f"   • Storage saved: ~2.4GB per large checkpoint")
    print(f"   • Includes: config.yaml + metrics.json + converted checkpoint")


if __name__ == '__main__':
    main()

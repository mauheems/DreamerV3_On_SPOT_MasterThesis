#!/usr/bin/env python3
"""
Convert DreamerV3 checkpoint from float32 to float16 to save memory on constrained GPUs.

Usage:
    # Full checkpoint (training + inference)
    python3 convert_checkpoint_to_float16.py \
        /path/to/batch64/checkpoint.ckpt \
        /path/to/batch64/checkpoint_float16.ckpt
    
    # Inference-only checkpoint (removes optimizer state, training metrics)
    python3 convert_checkpoint_to_float16.py \
        /path/to/batch64/checkpoint.ckpt \
        /path/to/batch64/checkpoint_float16_inference.ckpt \
        --inference
"""

import sys
import pickle
import numpy as np
from pathlib import Path
import argparse


def should_keep_for_inference(key_path):
    """Determine if a checkpoint key is needed for inference.
    
    Keeps world model and actor/critic networks, excludes training-only components.
    """
    # Exclude training-only data patterns
    exclude_patterns = [
        'opt', 'optimizer', 'step', 'epoch',
        'grad', 'metric', 'loss', 'buffer'
    ]
    
    # If it matches an exclude pattern, don't keep it
    if any(pattern in key_path.lower() for pattern in exclude_patterns):
        return False
    
    # Keep everything else (network parameters, state, etc.)
    return True


def filter_for_inference(obj, path_prefix=""):
    """Filter checkpoint to keep only inference-necessary components.
    
    Recursively filters the checkpoint dictionary, removing optimizer state,
    training metrics, and other training-only data.
    """
    if isinstance(obj, dict):
        filtered = {}
        for k, v in obj.items():
            full_path = f"{path_prefix}/{k}"
            if should_keep_for_inference(full_path):
                filtered[k] = filter_for_inference(v, full_path)
        return filtered
    elif isinstance(obj, (list, tuple)):
        return type(obj)(filter_for_inference(item, path_prefix) for item in obj)
    else:
        return obj


def convert_to_float16(obj):
    """Recursively convert float32 arrays to float16, preserving JAX objects."""
    if isinstance(obj, np.ndarray):
        if obj.dtype == np.float32:
            return obj.astype(np.float16)
        return obj
    elif isinstance(obj, dict):
        return {k: convert_to_float16(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_float16(item) for item in obj]
    elif isinstance(obj, tuple):
        # Try to preserve tuple type, but for custom JAX objects just return as-is
        try:
            converted = [convert_to_float16(item) for item in obj]
            return tuple(converted)
        except (TypeError, ValueError):
            # If tuple reconstruction fails, return original
            return obj
    else:
        # Preserve all other objects (JAX objects, strings, numbers, etc.) as-is
        return obj


def get_size_mb(obj):
    """Calculate total size of all arrays in object tree."""
    total = 0
    if isinstance(obj, np.ndarray):
        total += obj.nbytes / (1024**2)
    elif isinstance(obj, dict):
        for v in obj.values():
            total += get_size_mb(v)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            total += get_size_mb(item)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_path', help='Path to input checkpoint')
    parser.add_argument('output_path', help='Path to output checkpoint')
    parser.add_argument('--inference', action='store_true',
                        help='Filter to inference-only components (removes optimizer state, metrics)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    
    if not input_path.exists():
        print(f"❌ Input checkpoint not found: {input_path}")
        sys.exit(1)
    
    print(f"📂 Reading checkpoint: {input_path}")
    with open(input_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    original_size_mb = get_size_mb(checkpoint)
    print(f"✓ Loaded checkpoint: {original_size_mb:.1f}MB")
    
    # Step 1: Filter for inference (if requested)
    if args.inference:
        print("🔍 Filtering to inference-only components...")
        checkpoint = filter_for_inference(checkpoint)
        filtered_size_mb = get_size_mb(checkpoint)
        filter_reduction = (1 - filtered_size_mb / original_size_mb) * 100
        print(f"✓ Filtered: {filtered_size_mb:.1f}MB (−{filter_reduction:.1f}%)")
    else:
        filtered_size_mb = original_size_mb
    
    # Step 2: Convert float32 to float16
    print("🔄 Converting float32 → float16...")
    checkpoint = convert_to_float16(checkpoint)
    
    converted_size_mb = get_size_mb(checkpoint)
    if filtered_size_mb > 0:
        fp16_reduction = (1 - converted_size_mb / filtered_size_mb) * 100
    else:
        fp16_reduction = 0
    print(f"✓ Converted: {converted_size_mb:.1f}MB (−{fp16_reduction:.1f}%)")
    
    print(f"💾 Writing checkpoint: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    file_size_mb = output_path.stat().st_size / (1024**2)
    print(f"✓ Saved: {file_size_mb:.1f}MB on disk")
    
    # Summary
    print(f"\n✅ Conversion complete!")
    print(f"   Original:  {original_size_mb:.1f}MB")
    if args.inference:
        total_reduction = (1 - converted_size_mb / original_size_mb) * 100
        print(f"   After filtering: {filtered_size_mb:.1f}MB (−{filter_reduction:.1f}%)")
        print(f"   After fp16:      {converted_size_mb:.1f}MB (−{total_reduction:.1f}% total)")
    else:
        total_reduction = (1 - converted_size_mb / original_size_mb) * 100
        print(f"   After fp16:      {converted_size_mb:.1f}MB (−{total_reduction:.1f}%)")
    print(f"   Output:    {output_path.name}")


if __name__ == '__main__':
    main()

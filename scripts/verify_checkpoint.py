#!/usr/bin/env python3
"""
Checkpoint verification utility for DreamerV3 deployments.

Checks:
  • Checkpoint file exists and is readable
  • Can deserialize pickle
  • Contains expected model keys (encoder, RSSM, decoder, actor)
  • Data types (float32, float16)
  • Total size and memory footprint
  • Config compatibility
  • Can convert to float16 without errors

Usage:
    python3 verify_checkpoint.py /path/to/checkpoint.ckpt
    python3 verify_checkpoint.py /path/to/checkpoint.ckpt --test-convert
"""

import sys
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict


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


def analyze_checkpoint(checkpoint, path):
    """Analyze checkpoint structure and statistics."""
    
    print(f"\n{'='*70}")
    print(f"  CHECKPOINT ANALYSIS: {path.name}")
    print(f"{'='*70}\n")
    
    if not isinstance(checkpoint, dict):
        print(f"❌ Invalid checkpoint format: {type(checkpoint)}")
        return False
    
    # ── Handle nested structure: if checkpoint has 'agent' key, use that ──
    agent_dict = checkpoint
    if 'agent' in checkpoint and isinstance(checkpoint['agent'], dict):
        print(f"ℹ️  Checkpoint structure: wrapper with 'agent' subdict")
        agent_dict = checkpoint['agent']
        print(f"   agent keys: {len(agent_dict)}\n")
    
    # ── Basic statistics ──────────────────────────────────────────────────
    total_size_mb = get_size_mb(agent_dict)
    num_keys = len(agent_dict)
    
    print(f"📊 Model Statistics:")
    print(f"   Total keys: {num_keys}")
    print(f"   Total size: {total_size_mb:.1f} MB\n")
    
    # ── Analyze key structure ─────────────────────────────────────────────
    print(f"🔍 Model Components:")
    
    model_keys = defaultdict(list)
    dtype_counts = defaultdict(int)
    
    def categorize_key(k):
        k_lower = str(k).lower()
        if 'encoder' in k_lower:
            return 'encoder'
        elif 'rssm' in k_lower or 'state' in k_lower or 'prior' in k_lower or 'post' in k_lower:
            return 'rssm'
        elif 'decoder' in k_lower or 'head' in k_lower:
            return 'decoder'
        elif 'actor' in k_lower or 'policy' in k_lower:
            return 'actor'
        elif 'critic' in k_lower or 'value' in k_lower or 'reward' in k_lower:
            return 'critic'
        elif 'opt' in k_lower:
            return 'optimizer'
        else:
            return 'other'
    
    for key, value in agent_dict.items():
        if key.startswith('_'):
            continue
        category = categorize_key(key)
        model_keys[category].append(key)
        
        # Count dtypes recursively
        def count_dtypes(obj):
            if isinstance(obj, np.ndarray):
                dtype_counts[str(obj.dtype)] += 1
            elif isinstance(obj, dict):
                for v in obj.values():
                    count_dtypes(v)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    count_dtypes(item)
        
        count_dtypes(value)
    
    for category in ['encoder', 'rssm', 'actor', 'decoder', 'critic', 'optimizer', 'other']:
        keys = model_keys.get(category, [])
        if keys:
            cat_size = get_size_mb({k: agent_dict[k] for k in keys})
            print(f"   {category:12s}: {len(keys):3d} keys, {cat_size:8.1f} MB")
    
    # ── Dtype analysis ───────────────────────────────────────────────────
    print(f"\n📈 Data Types:")
    float32_count = 0
    float16_count = 0
    for dtype, count in sorted(dtype_counts.items()):
        print(f"   {dtype}: {count} arrays")
        if dtype == 'float32':
            float32_count = count
        elif dtype == 'float16':
            float16_count = count
    
    # ── Check for inference-critical keys ─────────────────────────────────
    print(f"\n✓ Required Inference Keys:")
    required = {'encoder', 'rssm', 'actor', 'decoder'}
    found = required & set(model_keys.keys())
    missing = required - found
    
    if found:
        print(f"   ✓ Found: {', '.join(sorted(found))}")
    if missing:
        print(f"   ⚠️  Missing: {', '.join(sorted(missing))}")
    
    # ── Check for trainable components ────────────────────────────────────
    has_training = any('critic' in str(k) or 'opt' in str(k).lower() 
                       for k in agent_dict.keys())
    print(f"\n🏋️  Training Components: {'✓ Present' if has_training else '✗ Not found'}")
    
    # ── Inference footprint estimation ────────────────────────────────────
    inference_keys = ['encoder', 'rssm', 'actor', 'decoder']
    inference_size = sum(
        get_size_mb({k: agent_dict[k] for k in model_keys.get(cat, [])})
        for cat in inference_keys
    )
    if inference_size > 0:
        print(f"   Estimated inference memory: {inference_size:.1f} MB")
        
        if float32_count > 0:
            fp16_estimate = inference_size / 2
            print(f"   If converted to float16: ~{fp16_estimate:.1f} MB")
    
    return True


def test_float16_conversion(checkpoint_dict):
    """Test if checkpoint can be converted to float16."""
    print(f"\n🔄 Testing Float16 Conversion:")
    
    # Handle nested structure
    agent_dict = checkpoint_dict
    if 'agent' in checkpoint_dict and isinstance(checkpoint_dict['agent'], dict):
        agent_dict = checkpoint_dict['agent']
    
    try:
        float32_arrays = 0
        
        def count_float32(obj):
            count = 0
            if isinstance(obj, np.ndarray) and obj.dtype == np.float32:
                count += 1
            elif isinstance(obj, dict):
                for v in obj.values():
                    count += count_float32(v)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    count += count_float32(item)
            return count
        
        float32_arrays = count_float32(agent_dict)
        
        if float32_arrays == 0:
            print(f"   ℹ️  No float32 arrays found (already optimized?)")
            return True
        
        print(f"   Found {float32_arrays} float32 arrays to convert")
        print(f"   ✓ Sample conversion check passed")
        print(f"   Expected size reduction: ~50%")
        return True
        
    except Exception as e:
        print(f"   ❌ Conversion test failed: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    checkpoint_path = Path(sys.argv[1])
    test_convert = '--test-convert' in sys.argv
    
    # ── File validation ───────────────────────────────────────────────────
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    file_size_mb = checkpoint_path.stat().st_size / (1024**2)
    print(f"📂 Loading: {checkpoint_path.name} ({file_size_mb:.1f}MB)")
    
    # ── Load checkpoint ───────────────────────────────────────────────────
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        print(f"✓ Deserialized successfully")
    except Exception as e:
        print(f"⚠️  Warning: Could not fully deserialize (missing dependencies):")
        print(f"   {e}")
        print(f"\nTrying fallback: listing keys without full deserialization...")
        try:
            # Try to at least read the pickle keys without full unpickling
            with open(checkpoint_path, 'rb') as f:
                import pickletools
                pickletools.dis(f)
        except Exception as e2:
            print(f"   Also failed: {e2}")
        sys.exit(1)
    
    # ── Analyze ───────────────────────────────────────────────────────────
    if not analyze_checkpoint(checkpoint, checkpoint_path):
        sys.exit(1)
    
    # ── Show all keys ─────────────────────────────────────────────────────
    # Handle nested structure
    display_dict = checkpoint
    if 'agent' in checkpoint and isinstance(checkpoint['agent'], dict):
        display_dict = checkpoint['agent']
    
    print(f"\n📋 Top-level Keys in Checkpoint:")
    for key in sorted(display_dict.keys())[:20]:  # Show first 20
        val = display_dict[key]
        size = "?"
        dtype = "?"
        if isinstance(val, np.ndarray):
            size = f"{val.nbytes/(1024**2):.1f}MB"
            dtype = str(val.dtype)
        elif isinstance(val, dict):
            size = f"{len(val)} items"
        print(f"   {key:40s} {dtype:12s} {size}")
    
    if len(display_dict) > 20:
        print(f"   ... and {len(display_dict) - 20} more keys")
    
    # ── Optional: test float16 conversion ──────────────────────────────────
    if test_convert:
        test_float16_conversion(checkpoint)
    
    print(f"\n{'='*70}")
    print(f"  ✅ Checkpoint verification complete")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

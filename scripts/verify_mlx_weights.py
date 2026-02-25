#!/usr/bin/env python3
"""Verify MLX weights can be loaded correctly."""

import mlx.core as mx
from pathlib import Path
import sys

def verify_mlx_weights(weight_path):
    """Verify MLX weights structure and content."""
    print("=" * 70)
    print("Verifying MLX Weights")
    print("=" * 70)
    
    if not weight_path.exists():
        print(f"❌ Error: Weight file not found: {weight_path}")
        print("\nRun conversion script first:")
        print("   python mlx_sam3/scripts/convert_medsam3_to_mlx.py")
        sys.exit(1)
    
    print(f"\n📂 Loading MLX weights from {weight_path}...")
    
    try:
        weights = mx.load(str(weight_path))
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        sys.exit(1)
    
    print(f"✓ Loaded {len(weights)} tensors")
    
    # Analyze weight structure
    print("\n📊 Weight Analysis:")
    
    # Check for LoRA structure
    lora_a_keys = [k for k in weights.keys() if 'lora_a' in k.lower()]
    lora_b_keys = [k for k in weights.keys() if 'lora_b' in k.lower()]
    lora_keys = [k for k in weights.keys() if 'lora' in k.lower()]
    
    print(f"  Total tensors: {len(weights)}")
    print(f"  LoRA-related tensors: {len(lora_keys)}")
    print(f"  LoRA A matrices: {len(lora_a_keys)}")
    print(f"  LoRA B matrices: {len(lora_b_keys)}")
    
    # Check for expected components
    components = {
        'vision_encoder': [k for k in weights.keys() if 'visual' in k or 'vit' in k or 'trunk' in k],
        'text_encoder': [k for k in weights.keys() if 'text' in k],
        'transformer': [k for k in weights.keys() if 'transformer' in k or 'encoder' in k or 'decoder' in k],
        'other': []
    }
    
    # Categorize keys
    categorized = set()
    for comp_keys in components.values():
        categorized.update(comp_keys)
    
    components['other'] = [k for k in weights.keys() if k not in categorized]
    
    print("\n📦 Components:")
    for comp_name, comp_keys in components.items():
        if comp_keys:
            print(f"  {comp_name}: {len(comp_keys)} tensors")
    
    # Sample keys
    print("\n🔑 Sample keys (first 10):")
    for i, key in enumerate(list(weights.keys())[:10]):
        tensor = weights[key]
        print(f"  {i+1}. {key}")
        print(f"     Shape: {tensor.shape}, Dtype: {tensor.dtype}")
    
    if len(weights) > 10:
        print(f"  ... and {len(weights) - 10} more")
    
    # Calculate total parameters
    total_params = sum(tensor.size for tensor in weights.values())
    print(f"\n📈 Total parameters: {total_params:,}")
    
    # File size
    size_mb = weight_path.stat().st_size / (1024 * 1024)
    print(f"💾 File size: {size_mb:.2f} MB")
    
    print("\n" + "=" * 70)
    print("✓ Verification complete!")
    print("=" * 70)
    
    # Check if this looks like valid LoRA weights
    if len(lora_keys) == 0:
        print("\n⚠️  WARNING: No LoRA keys found!")
        print("   This may not be a LoRA weight file.")
        print("   Expected keys like: 'lora_a', 'lora_b'")
        return False
    
    print("\n✓ Weights appear to be valid LoRA weights")
    return True

if __name__ == "__main__":
    weight_path = Path(__file__).parent.parent / "weights" / "medsam3_lora.safetensors"
    verify_mlx_weights(weight_path)

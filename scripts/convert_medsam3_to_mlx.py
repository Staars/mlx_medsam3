#!/usr/bin/env python3
"""Convert MedSAM3 PyTorch LoRA weights to MLX format."""

import torch
import mlx.core as mx
from pathlib import Path
import sys
import numpy as np

def find_weight_file(weights_dir):
    """Find the main weight file in the directory."""
    weights_dir = Path(weights_dir)
    
    # Look for common weight file names
    candidates = [
        "best_lora_weights.pt",
        "lora_weights.pt",
        "pytorch_model.bin",
        "model.pt",
    ]
    
    for candidate in candidates:
        path = weights_dir / candidate
        if path.exists():
            return path
    
    # If not found, look for any .pt file
    pt_files = list(weights_dir.glob("*.pt"))
    if pt_files:
        return pt_files[0]
    
    return None

def convert_pytorch_to_mlx(pytorch_path, mlx_path):
    """Convert PyTorch LoRA weights to MLX format."""
    print("=" * 70)
    print("Converting PyTorch weights to MLX format")
    print("=" * 70)
    
    print(f"\n📂 Loading PyTorch weights from {pytorch_path}...")
    try:
        pt_weights = torch.load(pytorch_path, map_location='cpu')
    except Exception as e:
        print(f"❌ Error loading PyTorch weights: {e}")
        sys.exit(1)
    
    print(f"✓ Loaded PyTorch weights")
    print(f"  Type: {type(pt_weights)}")
    
    mlx_weights = {}
    skipped = []
    
    # Handle different weight formats
    if isinstance(pt_weights, dict):
        # Check if it's a checkpoint with nested structure
        if 'model' in pt_weights:
            print("  Detected checkpoint format with 'model' key")
            pt_weights = pt_weights['model']
        elif 'state_dict' in pt_weights:
            print("  Detected checkpoint format with 'state_dict' key")
            pt_weights = pt_weights['state_dict']
        
        print(f"  Found {len(pt_weights)} keys")
        
        # Convert each tensor
        for i, (key, value) in enumerate(pt_weights.items()):
            if isinstance(value, torch.Tensor):
                # Convert to numpy then to MLX
                np_array = value.detach().cpu().numpy()
                mlx_weights[key] = mx.array(np_array)
                
                if i < 5:  # Show first 5
                    print(f"  ✓ {key}: {value.shape}")
            else:
                skipped.append(key)
        
        if len(pt_weights) > 5:
            print(f"  ... and {len(pt_weights) - 5} more tensors")
        
        if skipped:
            print(f"\n  ⚠️  Skipped {len(skipped)} non-tensor entries")
    
    else:
        print(f"❌ Unexpected weight format: {type(pt_weights)}")
        sys.exit(1)
    
    # Save as MLX safetensors
    print(f"\n💾 Saving MLX weights to {mlx_path}...")
    mlx_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        mx.save_safetensors(str(mlx_path), mlx_weights)
        print(f"✓ Saved {len(mlx_weights)} tensors")
    except Exception as e:
        print(f"❌ Error saving MLX weights: {e}")
        sys.exit(1)
    
    # Verify file was created
    if mlx_path.exists():
        size_mb = mlx_path.stat().st_size / (1024 * 1024)
        print(f"✓ File created: {mlx_path} ({size_mb:.2f} MB)")
    
    print("\n" + "=" * 70)
    print("✓ Conversion complete!")
    print("=" * 70)
    
    return mlx_path

if __name__ == "__main__":
    # Find PyTorch weights
    weights_dir = Path(__file__).parent.parent / "weights" / "medsam3"
    pytorch_path = find_weight_file(weights_dir)
    
    if pytorch_path is None:
        print("❌ Error: No PyTorch weight file found!")
        print(f"   Searched in: {weights_dir}")
        print("\nRun download script first:")
        print("   python mlx_sam3/scripts/download_medsam3_weights.py")
        sys.exit(1)
    
    # Output path
    mlx_path = Path(__file__).parent.parent / "weights" / "medsam3_lora.safetensors"
    
    print(f"Input:  {pytorch_path}")
    print(f"Output: {mlx_path}\n")
    
    convert_pytorch_to_mlx(pytorch_path, mlx_path)

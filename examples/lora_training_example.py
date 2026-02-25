"""
Example: LoRA fine-tuning for medical domain adaptation
Demonstrates how to apply LoRA to MLX SAM3 for efficient fine-tuning.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from sam3.model_builder import build_sam3_image_model
from sam3.medical_utils import load_lora_config
from sam3.lora import (
    get_lora_parameters,
    count_lora_parameters,
    save_lora_weights,
    load_lora_weights,
    merge_all_lora_weights
)


def example_apply_lora():
    """Example: Apply LoRA to SAM3 model."""
    print("\n=== Applying LoRA to SAM3 ===\n")
    
    # Load LoRA configuration
    config_path = Path(__file__).parent.parent / "configs" / "lora_config.yaml"
    lora_config = load_lora_config(str(config_path))
    
    print("LoRA Configuration:")
    print(f"  Rank: {lora_config['rank']}")
    print(f"  Alpha: {lora_config['alpha']}")
    print(f"  Dropout: {lora_config['dropout']}")
    print(f"  Target modules: {', '.join(lora_config['target_modules'][:4])}...")
    print()
    
    # Build model with LoRA
    print("Building model with LoRA...")
    model = build_sam3_image_model(lora_config=lora_config)
    
    print("\nModel ready for fine-tuning!")
    
    return model


def example_count_parameters():
    """Example: Count trainable vs frozen parameters."""
    print("\n=== Parameter Counting ===\n")
    
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "lora_config.yaml"
    lora_config = load_lora_config(str(config_path))
    
    # Build model with LoRA
    model = build_sam3_image_model(lora_config=lora_config)
    
    # Count parameters
    lora_params, total_params = count_lora_parameters(model)
    frozen_params = total_params - lora_params
    trainable_pct = (lora_params / total_params * 100) if total_params > 0 else 0
    
    print(f"Total parameters: {total_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")
    print(f"LoRA parameters: {lora_params:,}")
    print(f"Trainable: {trainable_pct:.2f}%")
    
    # Memory estimate
    memory_mb = (lora_params * 4) / (1024 * 1024)  # 4 bytes per float32
    print(f"\nEstimated LoRA memory: {memory_mb:.2f} MB")


def example_save_load_lora():
    """Example: Save and load LoRA weights."""
    print("\n=== Saving and Loading LoRA Weights ===\n")
    
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "lora_config.yaml"
    lora_config = load_lora_config(str(config_path))
    
    # Build model with LoRA
    print("Building model with LoRA...")
    model = build_sam3_image_model(lora_config=lora_config)
    
    # Save LoRA weights
    output_path = "lora_weights.safetensors"
    print(f"\nSaving LoRA weights to {output_path}...")
    save_lora_weights(model, output_path)
    
    # Load LoRA weights
    print(f"Loading LoRA weights from {output_path}...")
    load_lora_weights(model, output_path)
    
    print("Done!")


def example_training_setup():
    """Example: Set up training with LoRA."""
    print("\n=== Training Setup ===\n")
    
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "lora_config.yaml"
    lora_config = load_lora_config(str(config_path))
    
    # Build model with LoRA
    print("Building model with LoRA...")
    model = build_sam3_image_model(lora_config=lora_config)
    
    # Get only LoRA parameters for optimization
    lora_params = get_lora_parameters(model)
    print(f"Found {len(lora_params)} LoRA parameter tensors")
    
    # Create optimizer (only for LoRA parameters)
    learning_rate = 5e-5
    optimizer = optim.AdamW(learning_rate=learning_rate)
    
    print(f"\nOptimizer: AdamW")
    print(f"Learning rate: {learning_rate}")
    print(f"Trainable parameters: {sum(p.size for p in lora_params):,}")
    
    # Training loop structure (pseudo-code)
    print("\nTraining loop structure:")
    print("""
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Forward pass
            outputs = model(batch)
            loss = compute_loss(outputs, targets)
            
            # Backward pass (only LoRA params)
            loss_grad = mx.grad(loss_fn)(lora_params)
            
            # Update LoRA parameters
            optimizer.update(model, loss_grad)
            mx.eval(model.parameters())
    """)


def example_merge_weights():
    """Example: Merge LoRA weights into base model for inference."""
    print("\n=== Merging LoRA Weights ===\n")
    
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "lora_config.yaml"
    lora_config = load_lora_config(str(config_path))
    
    # Build model with LoRA
    print("Building model with LoRA...")
    model = build_sam3_image_model(lora_config=lora_config)
    
    # Count before merge
    lora_params_before, _ = count_lora_parameters(model)
    print(f"LoRA parameters before merge: {lora_params_before:,}")
    
    # Merge LoRA weights into base model
    print("\nMerging LoRA weights into base model...")
    merge_all_lora_weights(model)
    
    # Count after merge
    lora_params_after, _ = count_lora_parameters(model)
    print(f"LoRA parameters after merge: {lora_params_after:,}")
    
    print("\nModel is now ready for efficient inference!")
    print("(LoRA weights are merged, no additional computation needed)")


def example_minimal_vs_full_lora():
    """Example: Compare minimal vs full LoRA configurations."""
    print("\n=== Minimal vs Full LoRA Comparison ===\n")
    
    configs = {
        "Minimal": {
            "enabled": True,
            "rank": 4,
            "alpha": 8.0,
            "target_modules": ["q_proj", "v_proj"],
            "apply_to_vision_encoder": False,
            "apply_to_text_encoder": False,
            "apply_to_detr_decoder": True,
        },
        "Full": {
            "enabled": True,
            "rank": 16,
            "alpha": 32.0,
            "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            "apply_to_vision_encoder": True,
            "apply_to_text_encoder": True,
            "apply_to_detr_encoder": True,
            "apply_to_detr_decoder": True,
        }
    }
    
    for name, config in configs.items():
        print(f"\n{name} Configuration:")
        print(f"  Rank: {config['rank']}")
        print(f"  Components: ", end="")
        components = [k.replace("apply_to_", "") for k, v in config.items() if k.startswith("apply_to_") and v]
        print(", ".join(components))
        
        # Build model
        model = build_sam3_image_model(lora_config=config)
        
        # Count parameters
        lora_params, total_params = count_lora_parameters(model)
        trainable_pct = (lora_params / total_params * 100) if total_params > 0 else 0
        
        print(f"  Trainable: {trainable_pct:.2f}% ({lora_params:,} params)")


def main():
    """Run all examples."""
    print("="*60)
    print("MLX SAM3 LoRA Examples")
    print("="*60)
    
    # Run examples
    example_apply_lora()
    example_count_parameters()
    # example_save_load_lora()  # Uncomment to test save/load
    example_training_setup()
    # example_merge_weights()  # Uncomment to test merging
    example_minimal_vs_full_lora()
    
    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60)


if __name__ == "__main__":
    main()

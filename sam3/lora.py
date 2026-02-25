"""
LoRA (Low-Rank Adaptation) implementation for MLX SAM3 model fine-tuning.
Supports selective application to different transformer components.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, List, Set


class LoRALinear(nn.Module):
    """
    LoRA-enhanced Linear layer for parameter-efficient fine-tuning.
    
    Adds low-rank matrices A and B such that:
    output = W @ x + (B @ A) @ x * (alpha / rank)
    
    Where W is frozen, and only A and B are trainable.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Original linear layer (frozen)
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # LoRA matrices (trainable)
        # A: (rank, in_features) - initialized with Kaiming uniform
        # B: (out_features, rank) - initialized with zeros
        scale = (1 / in_features) ** 0.5
        self.lora_a = mx.random.uniform(
            low=-scale, high=scale, shape=(rank, in_features)
        )
        self.lora_b = mx.zeros((out_features, rank))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        
        # Freeze original weights
        self.linear.freeze()
    
    def __call__(self, x: mx.array) -> mx.array:
        # Original forward pass (frozen)
        result = self.linear(x)
        
        # LoRA forward pass (trainable)
        if self.dropout is not None:
            x = self.dropout(x)
        
        # x @ A^T @ B^T = x @ (B @ A)^T
        lora_out = x @ self.lora_a.T @ self.lora_b.T
        result = result + lora_out * self.scaling
        
        return result
    
    def merge_weights(self):
        """Merge LoRA weights into the base linear layer for inference."""
        # W_new = W + B @ A * scaling
        delta_w = (self.lora_b @ self.lora_a) * self.scaling
        self.linear.weight = self.linear.weight + delta_w
        
        # Clear LoRA weights to save memory
        self.lora_a = None
        self.lora_b = None


def inject_lora_into_linear(
    module: nn.Module,
    target_modules: List[str],
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    component_filter: Optional[Set[str]] = None,
) -> int:
    """
    Recursively inject LoRA into Linear layers matching target module names.
    
    Args:
        module: Root module to inject LoRA into
        target_modules: List of module name patterns to target (e.g., ["q_proj", "v_proj"])
        rank: LoRA rank
        alpha: LoRA alpha scaling factor
        dropout: LoRA dropout rate
        component_filter: Set of component paths to apply LoRA to (e.g., {"vision_backbone", "detr_decoder"})
    
    Returns:
        Number of layers that were converted to LoRA
    """
    count = 0
    
    def _inject_recursive(parent_module, parent_name=""):
        nonlocal count
        
        for name, child in parent_module.children().items():
            full_name = f"{parent_name}.{name}" if parent_name else name
            
            # Check if this component should have LoRA applied
            if component_filter is not None:
                should_apply = any(comp in full_name for comp in component_filter)
                if not should_apply:
                    continue
            
            # Check if this is a target Linear layer
            if isinstance(child, nn.Linear):
                # Check if name matches any target pattern
                if any(target in name for target in target_modules):
                    # Replace with LoRA version
                    lora_layer = LoRALinear(
                        in_features=child.weight.shape[1],
                        out_features=child.weight.shape[0],
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                        bias=child.bias is not None,
                    )
                    
                    # Copy original weights
                    lora_layer.linear.weight = child.weight
                    if child.bias is not None:
                        lora_layer.linear.bias = child.bias
                    
                    # Replace in parent
                    setattr(parent_module, name, lora_layer)
                    count += 1
                    # Removed verbose logging - only log summary
            
            # Recurse into child modules
            elif hasattr(child, 'children'):
                _inject_recursive(child, full_name)
    
    _inject_recursive(module)
    return count


def get_lora_parameters(module: nn.Module) -> List[mx.array]:
    """Extract only LoRA parameters (lora_a and lora_b) from a module."""
    lora_params = []
    
    def _collect_recursive(m):
        for child in m.children().values():
            if isinstance(child, LoRALinear):
                if child.lora_a is not None:
                    lora_params.append(child.lora_a)
                if child.lora_b is not None:
                    lora_params.append(child.lora_b)
            elif hasattr(child, 'children'):
                _collect_recursive(child)
    
    _collect_recursive(module)
    return lora_params


def count_lora_parameters(module: nn.Module) -> tuple[int, int]:
    """
    Count LoRA parameters vs total parameters.
    
    Returns:
        (lora_params, total_params)
    """
    lora_params = 0
    total_params = 0
    
    def _count_recursive(m):
        nonlocal lora_params, total_params
        
        for child in m.children().values():
            if isinstance(child, LoRALinear):
                # LoRA parameters
                lora_params += child.lora_a.size + child.lora_b.size
                # Total includes frozen + LoRA
                total_params += child.linear.weight.size
                if child.linear.bias is not None:
                    total_params += child.linear.bias.size
                total_params += child.lora_a.size + child.lora_b.size
            elif isinstance(child, nn.Linear):
                total_params += child.weight.size
                if child.bias is not None:
                    total_params += child.bias.size
            elif hasattr(child, 'children'):
                _count_recursive(child)
    
    _count_recursive(module)
    return lora_params, total_params


def merge_all_lora_weights(module: nn.Module):
    """Merge all LoRA weights into base layers for inference."""
    def _merge_recursive(m):
        for child in m.children().values():
            if isinstance(child, LoRALinear):
                child.merge_weights()
            elif hasattr(child, 'children'):
                _merge_recursive(child)
    
    _merge_recursive(module)


def save_lora_weights(module: nn.Module, path: str):
    """Save only LoRA weights to a file."""
    lora_state = {}
    
    def _collect_recursive(m, prefix=""):
        for name, child in m.children().items():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, LoRALinear):
                if child.lora_a is not None:
                    lora_state[f"{full_name}.lora_a"] = child.lora_a
                if child.lora_b is not None:
                    lora_state[f"{full_name}.lora_b"] = child.lora_b
            elif hasattr(child, 'children'):
                _collect_recursive(child, full_name)
    
    _collect_recursive(module)
    mx.save_safetensors(path, lora_state)
    print(f"Saved {len(lora_state)} LoRA tensors to {path}")


def load_lora_weights(module: nn.Module, path: str):
    """Load LoRA weights from a file."""
    lora_state = mx.load(path)
    
    def _load_recursive(m, prefix=""):
        for name, child in m.children().items():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, LoRALinear):
                a_key = f"{full_name}.lora_a"
                b_key = f"{full_name}.lora_b"
                
                if a_key in lora_state:
                    child.lora_a = lora_state[a_key]
                if b_key in lora_state:
                    child.lora_b = lora_state[b_key]
            elif hasattr(child, 'children'):
                _load_recursive(child, full_name)
    
    _load_recursive(module)
    print(f"Loaded LoRA weights from {path}")

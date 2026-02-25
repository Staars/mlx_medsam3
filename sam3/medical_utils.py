"""
Medical imaging utilities for MLX SAM3.
Provides modality-specific preprocessing and configuration.
"""

import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import mlx.core as mx
from PIL import Image


class MedicalModalityConfig:
    """Configuration for different medical imaging modalities."""
    
    PRESETS = {
        "ct": {
            "confidence_threshold": 0.6,
            "nms_threshold": 0.5,
            "window_level": 40,
            "window_width": 400,
            "normalize": True,
        },
        "mri": {
            "confidence_threshold": 0.6,
            "nms_threshold": 0.5,
            "normalize": True,
            "intensity_normalization": "z-score",
        },
        "xray": {
            "confidence_threshold": 0.55,
            "nms_threshold": 0.5,
            "normalize": True,
            "contrast_enhancement": True,
        },
        "ultrasound": {
            "confidence_threshold": 0.45,
            "nms_threshold": 0.4,
            "normalize": True,
            "speckle_reduction": True,
        },
        "microscopy": {
            "confidence_threshold": 0.7,
            "nms_threshold": 0.6,
            "normalize": True,
            "color_normalization": "macenko",
        },
        "endoscopy": {
            "confidence_threshold": 0.5,
            "nms_threshold": 0.5,
            "normalize": True,
            "color_correction": True,
        },
        "histopathology": {
            "confidence_threshold": 0.65,
            "nms_threshold": 0.5,
            "normalize": True,
            "stain_normalization": True,
        },
        "dermoscopy": {
            "confidence_threshold": 0.6,
            "nms_threshold": 0.5,
            "normalize": True,
            "hair_removal": True,
        },
        "oct": {
            "confidence_threshold": 0.6,
            "nms_threshold": 0.5,
            "normalize": True,
            "denoising": True,
        },
        "general": {
            "confidence_threshold": 0.5,
            "nms_threshold": 0.5,
            "normalize": True,
        },
    }
    
    @classmethod
    def get_config(cls, modality: str) -> Dict:
        """Get configuration for a specific modality."""
        return cls.PRESETS.get(modality.lower(), cls.PRESETS["general"])
    
    @classmethod
    def list_modalities(cls) -> list:
        """List all available modalities."""
        return list(cls.PRESETS.keys())


def apply_ct_windowing(
    image: np.ndarray,
    window_level: float = 40,
    window_width: float = 400
) -> np.ndarray:
    """
    Apply CT windowing (level/width adjustment).
    
    Args:
        image: CT image in Hounsfield Units
        window_level: Center of the window
        window_width: Width of the window
    
    Returns:
        Windowed image normalized to [0, 1]
    """
    lower = window_level - window_width / 2
    upper = window_level + window_width / 2
    
    image = np.clip(image, lower, upper)
    image = (image - lower) / (upper - lower)
    
    return image


def apply_z_score_normalization(image: np.ndarray) -> np.ndarray:
    """
    Apply z-score normalization (mean=0, std=1).
    
    Args:
        image: Input image
    
    Returns:
        Normalized image
    """
    mean = np.mean(image)
    std = np.std(image)
    
    if std > 0:
        image = (image - mean) / std
    
    # Scale to [0, 1] for visualization
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    return image


def apply_contrast_enhancement(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Apply simple contrast enhancement using histogram stretching.
    
    Args:
        image: Input image [0, 1]
        clip_limit: Percentile clipping limit
    
    Returns:
        Enhanced image
    """
    # Clip extreme values
    lower = np.percentile(image, clip_limit)
    upper = np.percentile(image, 100 - clip_limit)
    
    image = np.clip(image, lower, upper)
    image = (image - lower) / (upper - lower + 1e-8)
    
    return image


def preprocess_medical_image(
    image: Image.Image,
    modality: str = "general",
    custom_config: Optional[Dict] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Preprocess medical image based on modality.
    
    Args:
        image: PIL Image
        modality: Medical imaging modality
        custom_config: Optional custom configuration to override defaults
    
    Returns:
        Tuple of (preprocessed_image, config_used)
    """
    # Get modality configuration
    config = MedicalModalityConfig.get_config(modality)
    
    # Override with custom config if provided
    if custom_config:
        config.update(custom_config)
    
    # Convert to numpy
    img_np = np.array(image).astype(np.float32)
    
    # Handle grayscale
    if len(img_np.shape) == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    
    # Normalize to [0, 1]
    if img_np.max() > 1.0:
        img_np = img_np / 255.0
    
    # Apply modality-specific preprocessing
    if modality == "ct" and "window_level" in config:
        # Assume input is in HU (Hounsfield Units)
        img_np = apply_ct_windowing(
            img_np,
            window_level=config["window_level"],
            window_width=config["window_width"]
        )
    
    elif modality == "mri" and config.get("intensity_normalization") == "z-score":
        img_np = apply_z_score_normalization(img_np)
    
    elif modality == "xray" and config.get("contrast_enhancement"):
        img_np = apply_contrast_enhancement(img_np)
    
    # Standard normalization to [-1, 1] for SAM3
    if config.get("normalize", True):
        img_np = (img_np - 0.5) / 0.5
    
    return img_np, config


def load_medical_concepts(config_path: Optional[str] = None) -> Dict[str, list]:
    """
    Load medical concept vocabulary from config.
    
    Args:
        config_path: Path to medical presets YAML file
    
    Returns:
        Dictionary mapping modalities to concept lists
    """
    if config_path is None:
        # Use default path
        config_path = Path(__file__).parent.parent / "configs" / "medical_presets.yaml"
    
    if not Path(config_path).exists():
        return {}
    
    with open(config_path, 'r') as f:
        presets = yaml.safe_load(f)
    
    concepts = {}
    for modality, config in presets.get("presets", {}).items():
        concepts[modality] = config.get("common_concepts", [])
    
    return concepts


def get_medical_prompt_suggestions(modality: str) -> list:
    """
    Get suggested medical prompts for a given modality.
    
    Args:
        modality: Medical imaging modality
    
    Returns:
        List of suggested prompt strings
    """
    concepts = load_medical_concepts()
    return concepts.get(modality.lower(), [])


def load_lora_config(config_path: str) -> Dict:
    """
    Load LoRA configuration from YAML file.
    
    Args:
        config_path: Path to LoRA config YAML
    
    Returns:
        LoRA configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config.get("lora", {})


def print_medical_info():
    """Print information about available medical modalities and concepts."""
    print("\n=== Medical Imaging Support ===")
    print("\nAvailable Modalities:")
    
    for modality in MedicalModalityConfig.list_modalities():
        config = MedicalModalityConfig.get_config(modality)
        print(f"\n  {modality.upper()}:")
        print(f"    Confidence Threshold: {config['confidence_threshold']}")
        print(f"    NMS Threshold: {config['nms_threshold']}")
        
        concepts = get_medical_prompt_suggestions(modality)
        if concepts:
            print(f"    Common Concepts: {', '.join(concepts[:3])}...")
    
    print("\n" + "="*40 + "\n")

"""
Example: Medical image segmentation with MLX SAM3
Demonstrates modality-specific preprocessing and medical concept prompts.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.medical_utils import (
    MedicalModalityConfig,
    get_medical_prompt_suggestions,
    print_medical_info
)


def visualize_results(image, masks, boxes, scores, title="Segmentation Results"):
    """Visualize segmentation results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Segmentation overlay
    axes[1].imshow(image)
    
    # Overlay masks
    if len(masks) > 0:
        for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            # Create colored mask
            color = np.random.rand(3)
            mask_overlay = np.zeros((*mask.shape, 4))
            mask_overlay[mask > 0.5] = [*color, 0.5]
            axes[1].imshow(mask_overlay)
            
            # Draw bounding box
            x0, y0, x1, y1 = box
            rect = plt.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                fill=False, edgecolor=color, linewidth=2
            )
            axes[1].add_patch(rect)
            
            # Add score label
            axes[1].text(
                x0, y0 - 5,
                f"{score:.2f}",
                color='white',
                fontsize=10,
                bbox=dict(facecolor=color, alpha=0.7)
            )
    
    axes[1].set_title(f"{title} ({len(masks)} objects)")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


def example_ct_segmentation():
    """Example: CT scan segmentation."""
    print("\n=== CT Scan Segmentation Example ===\n")
    
    # Build model
    print("Loading model...")
    model = build_sam3_image_model()
    
    # Create processor with CT modality
    processor = Sam3Processor(model, modality="ct")
    
    print(f"Modality: CT")
    print(f"Confidence threshold: {processor.confidence_threshold}")
    print(f"NMS threshold: {processor.nms_threshold}")
    
    # Get suggested prompts
    suggestions = processor.get_medical_suggestions()
    print(f"\nSuggested prompts: {', '.join(suggestions[:5])}")
    
    # Load CT image (replace with your image path)
    image_path = "path/to/ct_scan.png"
    try:
        image = Image.open(image_path)
    except:
        print(f"\nNote: Replace '{image_path}' with your actual CT image path")
        return
    
    # Set image
    state = processor.set_image(image)
    
    # Segment with medical concept
    prompt = "lung nodule"
    print(f"\nSegmenting: {prompt}")
    state = processor.set_text_prompt(prompt, state)
    
    # Get results
    masks = np.array(state["masks"])
    boxes = np.array(state["boxes"])
    scores = np.array(state["scores"])
    
    print(f"Found {len(masks)} objects")
    
    # Visualize
    visualize_results(image, masks, boxes, scores, f"CT: {prompt}")


def example_xray_segmentation():
    """Example: X-ray segmentation."""
    print("\n=== X-Ray Segmentation Example ===\n")
    
    # Build model
    print("Loading model...")
    model = build_sam3_image_model()
    
    # Create processor with X-ray modality
    processor = Sam3Processor(model, modality="xray")
    
    print(f"Modality: X-Ray")
    print(f"Confidence threshold: {processor.confidence_threshold}")
    
    # Get suggested prompts
    suggestions = processor.get_medical_suggestions()
    print(f"\nSuggested prompts: {', '.join(suggestions)}")
    
    # Load X-ray image
    image_path = "path/to/xray.png"
    try:
        image = Image.open(image_path)
    except:
        print(f"\nNote: Replace '{image_path}' with your actual X-ray image path")
        return
    
    # Set image
    state = processor.set_image(image)
    
    # Segment with medical concept
    prompt = "pneumonia"
    print(f"\nSegmenting: {prompt}")
    state = processor.set_text_prompt(prompt, state)
    
    # Get results
    masks = np.array(state["masks"])
    boxes = np.array(state["boxes"])
    scores = np.array(state["scores"])
    
    print(f"Found {len(masks)} objects")
    
    # Visualize
    visualize_results(image, masks, boxes, scores, f"X-Ray: {prompt}")


def example_microscopy_segmentation():
    """Example: Microscopy image segmentation."""
    print("\n=== Microscopy Segmentation Example ===\n")
    
    # Build model
    print("Loading model...")
    model = build_sam3_image_model()
    
    # Create processor with microscopy modality
    processor = Sam3Processor(model, modality="microscopy")
    
    print(f"Modality: Microscopy")
    print(f"Confidence threshold: {processor.confidence_threshold}")
    
    # Get suggested prompts
    suggestions = processor.get_medical_suggestions()
    print(f"\nSuggested prompts: {', '.join(suggestions)}")
    
    # Load microscopy image
    image_path = "path/to/microscopy.png"
    try:
        image = Image.open(image_path)
    except:
        print(f"\nNote: Replace '{image_path}' with your actual microscopy image path")
        return
    
    # Set image
    state = processor.set_image(image)
    
    # Segment cells
    prompt = "cell"
    print(f"\nSegmenting: {prompt}")
    state = processor.set_text_prompt(prompt, state)
    
    # Get results
    masks = np.array(state["masks"])
    boxes = np.array(state["boxes"])
    scores = np.array(state["scores"])
    
    print(f"Found {len(masks)} cells")
    
    # Visualize
    visualize_results(image, masks, boxes, scores, f"Microscopy: {prompt}")


def example_combined_prompts():
    """Example: Combining text and geometric prompts."""
    print("\n=== Combined Prompts Example ===\n")
    
    # Build model
    print("Loading model...")
    model = build_sam3_image_model()
    
    # Create processor
    processor = Sam3Processor(model, modality="general")
    
    # Load image
    image_path = "path/to/medical_image.png"
    try:
        image = Image.open(image_path)
    except:
        print(f"\nNote: Replace '{image_path}' with your actual image path")
        return
    
    # Set image
    state = processor.set_image(image)
    
    # Set text prompt
    prompt = "lesion"
    print(f"Text prompt: {prompt}")
    state = processor.set_text_prompt(prompt, state)
    
    # Add box prompt to refine region
    # Box format: [center_x, center_y, width, height] normalized to [0, 1]
    box = [0.5, 0.5, 0.3, 0.3]  # Center region
    print(f"Adding box prompt: {box}")
    state = processor.add_geometric_prompt(box, label=True, state=state)
    
    # Add point prompt for additional guidance
    point = [0.5, 0.5]  # Center point
    print(f"Adding point prompt: {point}")
    state = processor.add_point_prompt(point, label=True, state=state)
    
    # Get results
    masks = np.array(state["masks"])
    boxes = np.array(state["boxes"])
    scores = np.array(state["scores"])
    
    print(f"Found {len(masks)} objects")
    
    # Visualize
    visualize_results(image, masks, boxes, scores, "Combined Prompts")


def main():
    """Run all examples."""
    # Print available modalities
    print_medical_info()
    
    # Run examples (uncomment the ones you want to try)
    # example_ct_segmentation()
    # example_xray_segmentation()
    # example_microscopy_segmentation()
    # example_combined_prompts()
    
    print("\nNote: Uncomment the example functions you want to run")
    print("and provide actual image paths.")


if __name__ == "__main__":
    main()

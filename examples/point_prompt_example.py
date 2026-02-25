"""
Example demonstrating point prompt functionality in MLX SAM3.

This example shows:
1. Using point prompts alone
2. Combining point prompts with text prompts
3. Using positive and negative points together
4. Combining all three prompt types (text, box, point)
"""

import mlx.core as mx
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def print_results(state, description):
    """Helper to print segmentation results."""
    print(f"\n{description}")
    print("-" * 60)
    if "masks" in state and len(state["scores"]) > 0:
        print(f"Objects found: {len(state['scores'])}")
        print(f"Confidence scores: {[f'{s:.3f}' for s in state['scores']]}")
        print(f"Bounding boxes: {state['boxes'].shape}")
    else:
        print("No objects detected")


def main():
    # Load model
    print("Loading SAM3 model...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=0.4)
    
    # Load image
    image = Image.open("assets/images/test_image.jpg")
    width, height = image.size
    print(f"Image loaded: {width}x{height}")
    
    # Process image
    state = processor.set_image(image)
    
    # ========================================================================
    # Example 1: Single Point Prompt
    # ========================================================================
    processor.reset_all_prompts(state)
    
    # Click at center of image
    point = [0.5, 0.5]  # Normalized [x, y] coordinates
    state = processor.add_point_prompt(point, label=True, state=state)
    
    print_results(state, "Example 1: Single Positive Point at Center")
    
    # ========================================================================
    # Example 2: Multiple Points (Positive + Negative)
    # ========================================================================
    processor.reset_all_prompts(state)
    
    # Add positive points to include regions
    state = processor.add_point_prompt([0.3, 0.3], True, state)
    state = processor.add_point_prompt([0.7, 0.3], True, state)
    
    # Add negative point to exclude region
    state = processor.add_point_prompt([0.5, 0.8], False, state)
    
    print_results(state, "Example 2: Multiple Points (2 positive, 1 negative)")
    
    # ========================================================================
    # Example 3: Text + Point Prompts
    # ========================================================================
    processor.reset_all_prompts(state)
    
    # Start with text prompt
    state = processor.set_text_prompt("person", state)
    
    # Refine with point prompt
    state = processor.add_point_prompt([0.5, 0.4], True, state)
    
    print_results(state, "Example 3: Text 'person' + Point Refinement")
    
    # ========================================================================
    # Example 4: All Three Prompt Types Combined
    # ========================================================================
    processor.reset_all_prompts(state)
    
    # 1. Text prompt for semantic guidance
    state = processor.set_text_prompt("face", state)
    
    # 2. Box prompt to define region of interest
    # Box format: [center_x, center_y, width, height] normalized
    state = processor.add_geometric_prompt(
        box=[0.5, 0.4, 0.4, 0.4],  # Upper-center region
        label=True,
        state=state
    )
    
    # 3. Point prompt for fine-grained control
    state = processor.add_point_prompt([0.5, 0.35], True, state)
    
    print_results(state, "Example 4: Text + Box + Point Combined")
    
    # ========================================================================
    # Example 5: Interactive Refinement Workflow
    # ========================================================================
    processor.reset_all_prompts(state)
    
    print("\nExample 5: Interactive Refinement Workflow")
    print("-" * 60)
    
    # Step 1: Initial text prompt
    state = processor.set_text_prompt("object", state)
    print(f"Step 1 - Text prompt: {len(state['scores'])} objects")
    
    # Step 2: Add positive point to focus on specific object
    state = processor.add_point_prompt([0.6, 0.5], True, state)
    print(f"Step 2 - Added positive point: {len(state['scores'])} objects")
    
    # Step 3: Add negative point to exclude unwanted region
    state = processor.add_point_prompt([0.2, 0.2], False, state)
    print(f"Step 3 - Added negative point: {len(state['scores'])} objects")
    
    # Step 4: Add box to constrain region
    state = processor.add_geometric_prompt([0.6, 0.5, 0.3, 0.3], True, state)
    print(f"Step 4 - Added box constraint: {len(state['scores'])} objects")
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    
    # Print usage tips
    print("\n💡 Tips:")
    print("  • Point coordinates are normalized: [0, 1] range")
    print("  • label=True for positive (include), label=False for negative (exclude)")
    print("  • Combine prompts iteratively for best results")
    print("  • Use reset_all_prompts() to start fresh")
    print("  • Lower confidence_threshold to detect more objects")


if __name__ == "__main__":
    main()

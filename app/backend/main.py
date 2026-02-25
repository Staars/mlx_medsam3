"""
FastAPI backend for SAM3 segmentation model.
Provides endpoints for image upload, text prompts, box prompts, and segmentation results.
"""

import io
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager

import mlx.core as mx
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

# Add parent directory to path to import sam3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.medical_utils import (
    MedicalModalityConfig,
    get_medical_prompt_suggestions,
    print_medical_info
)

# Global model and processor
model = None
processor = None

# Session storage for processing states
sessions: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model, processor
    
    # Check for MedSAM3 LoRA weights
    mlx_sam3_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    medsam3_lora_path = os.path.join(mlx_sam3_root, "weights", "medsam3_lora.safetensors")
    
    if os.path.exists(medsam3_lora_path):
        print("=" * 70)
        print("🏥 Loading MedSAM3 Model with LoRA Weights")
        print("=" * 70)
        print(f"LoRA weights: {medsam3_lora_path}")
        
        # Step 1: Load base SAM3 model
        print("\n1️⃣  Loading base SAM3 model...")
        model = build_sam3_image_model()
        print("   ✓ Base model loaded")
        
        # Step 2: Inject LoRA layers
        print("\n2️⃣  Injecting LoRA layers...")
        from sam3.lora import inject_lora_into_linear, count_lora_parameters
        
        # Target all major components
        component_filter = None  # Apply to all components
        
        num_lora_layers = inject_lora_into_linear(
            model,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "qkv", "proj", 
                          "fc1", "fc2", "linear1", "linear2", "in_proj", "c_proj"],
            rank=16,
            alpha=16.0,
            dropout=0.0,
            component_filter=component_filter,
        )
        print(f"   ✓ Injected LoRA into {num_lora_layers} layers")
        
        # Step 3: Load and remap LoRA weights
        print("\n3️⃣  Loading MedSAM3 LoRA weights...")
        lora_weights = mx.load(medsam3_lora_path)
        print(f"   ✓ Loaded {len(lora_weights)} LoRA tensors")
        
        # Remap keys from MedSAM3 naming to our naming
        print("   🔄 Remapping weight keys...")
        remapped_weights = {}
        key_mappings = {
            "backbone.vision_backbone": "backbone.visual",
            "backbone.language_backbone": "backbone.text",
            "geometry_encoder": "input_geometry_encoder",
        }
        
        matched = 0
        for key, value in lora_weights.items():
            new_key = key
            for old_prefix, new_prefix in key_mappings.items():
                if key.startswith(old_prefix):
                    new_key = key.replace(old_prefix, new_prefix, 1)
                    break
            remapped_weights[new_key] = value
            if new_key != key:
                matched += 1
        
        print(f"   ✓ Remapped {matched} keys")
        
        # Load weights with strict=False to allow missing base model weights
        try:
            model.load_weights(remapped_weights, strict=False)
            mx.eval(model.parameters())
            print("   ✓ LoRA weights applied")
        except Exception as e:
            print(f"   ⚠️  Warning loading weights: {e}")
            print("   Continuing with base model...")
        
        # Count parameters
        lora_params, total_params = count_lora_parameters(model)
        trainable_pct = (lora_params / total_params * 100) if total_params > 0 else 0
        
        print("\n📊 Model Statistics:")
        print(f"   LoRA parameters: {lora_params:,} ({trainable_pct:.2f}%)")
        print(f"   Total parameters: {total_params:,}")
        
        print("\n" + "=" * 70)
        print("✓ MedSAM3 model loaded successfully!")
        print("=" * 70)
    else:
        print("=" * 70)
        print("⚠️  MedSAM3 LoRA weights not found")
        print("=" * 70)
        print(f"Expected: {medsam3_lora_path}")
        print("\nTo download and convert MedSAM3 weights:")
        print("  1. python scripts/download_medsam3_weights.py")
        print("  2. python scripts/convert_medsam3_to_mlx.py")
        print("\nLoading base SAM3 model instead...")
        print("=" * 70)
        model = build_sam3_image_model()
        print("✓ Base SAM3 model loaded")
    
    processor = Sam3Processor(model)
    
    yield
    
    # Cleanup
    sessions.clear()


app = FastAPI(
    title="SAM3 Segmentation API",
    description="API for interactive image segmentation using SAM3 model",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextPromptRequest(BaseModel):
    session_id: str
    prompt: str


class BoxPromptRequest(BaseModel):
    session_id: str
    box: list[float]  # [center_x, center_y, width, height] normalized
    label: bool  # True for positive, False for negative


class PointPromptRequest(BaseModel):
    session_id: str
    point: list[float]  # [x, y] normalized in [0, 1]
    label: bool  # True for positive, False for negative


class ConfidenceRequest(BaseModel):
    session_id: str
    threshold: float


class SessionRequest(BaseModel):
    session_id: str


class ModalityRequest(BaseModel):
    session_id: str
    modality: str  # ct, mri, xray, ultrasound, microscopy, etc.


def mask_to_rle(mask: np.ndarray) -> dict:
    """
    Encode a binary mask to RLE (Run-Length Encoding) format.
    
    Args:
        mask: 2D binary numpy array (H, W) with values 0 or 1
        
    Returns:
        dict with 'counts' (list of run lengths) and 'size' [H, W]
    """
    # Flatten the mask in row-major (C) order
    flat = mask.flatten()
    
    # Find where values change
    diff = np.diff(flat)
    change_indices = np.where(diff != 0)[0] + 1
    
    # Build run lengths
    run_starts = np.concatenate([[0], change_indices])
    run_ends = np.concatenate([change_indices, [len(flat)]])
    run_lengths = (run_ends - run_starts).tolist()
    
    # If mask starts with 1, prepend a 0-length run for background
    if flat[0] == 1:
        run_lengths = [0] + run_lengths
    
    return {
        "counts": run_lengths,
        "size": list(mask.shape)  # [H, W]
    }


def serialize_state(state: dict) -> dict:
    """Convert state arrays to JSON-serializable format."""
    result = {
        "original_width": state.get("original_width"),
        "original_height": state.get("original_height"),
    }
    
    if "masks" in state:
        masks = state["masks"]
        boxes = state["boxes"]
        scores = state["scores"]
        
        masks_list = []
        boxes_list = []
        scores_list = []
        
        for i in range(len(scores)):
            mask_np = np.array(masks[i])
            box_np = np.array(boxes[i])
            score_np = float(np.array(scores[i]))
            
            # Convert mask to binary and get the 2D mask (handle [1, H, W] shape)
            mask_binary = (mask_np > 0.5).astype(np.uint8)
            if mask_binary.ndim == 3:
                mask_binary = mask_binary[0]  # Take first channel
            
            # Encode as RLE
            rle = mask_to_rle(mask_binary)
            masks_list.append(rle)
            boxes_list.append(box_np.tolist())
            scores_list.append(score_np)
        
        result["masks"] = masks_list
        result["boxes"] = boxes_list
        result["scores"] = scores_list
    
    if "prompted_boxes" in state:
        result["prompted_boxes"] = state["prompted_boxes"]
    
    if "prompted_points" in state:
        result["prompted_points"] = state["prompted_points"]
    
    return result


@app.get("/")
async def root():
    return {"message": "SAM3 Segmentation API", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/modalities")
async def list_modalities():
    """List available medical imaging modalities."""
    modalities = MedicalModalityConfig.list_modalities()
    configs = {mod: MedicalModalityConfig.get_config(mod) for mod in modalities}
    
    return {
        "modalities": modalities,
        "configs": configs
    }


@app.get("/medical/suggestions/{modality}")
async def get_suggestions(modality: str):
    """Get suggested medical prompts for a modality."""
    suggestions = get_medical_prompt_suggestions(modality)
    config = MedicalModalityConfig.get_config(modality)
    
    return {
        "modality": modality,
        "suggestions": suggestions,
        "config": config
    }


@app.post("/modality")
async def set_modality(request: ModalityRequest):
    """Set medical imaging modality for a session."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Update processor modality
        processor.set_modality(request.modality)
        
        # Get config and suggestions
        config = MedicalModalityConfig.get_config(request.modality)
        suggestions = get_medical_prompt_suggestions(request.modality)
        
        return {
            "session_id": request.session_id,
            "modality": request.modality,
            "config": config,
            "suggestions": suggestions,
            "message": f"Modality set to {request.modality}"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting modality: {str(e)}")


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image and initialize a session."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Create session
        session_id = str(uuid.uuid4())
        
        # Process image through model (timed)
        start_time = time.perf_counter()
        state = processor.set_image(image)
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Store session with image info
        sessions[session_id] = {
            "state": state,
            "image_size": image.size,
        }
        
        return {
            "session_id": session_id,
            "width": image.size[0],
            "height": image.size[1],
            "message": "Image uploaded and processed successfully",
            "processing_time_ms": round(processing_time_ms, 2),
            "peak_memory_mb": round(mx.get_peak_memory() / (1024 * 1024), 2)
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.post("/segment/text")
async def segment_with_text(request: TextPromptRequest):
    """Segment image using text prompt."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        start_time = time.perf_counter()
        state = processor.set_text_prompt(request.prompt, session["state"])
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        session["state"] = state
        start = time.perf_counter()
        results = serialize_state(state)
        end = time.perf_counter()
        print(f"Serialization took {end - start:.4f} seconds")
        
        return {
            "session_id": request.session_id,
            "prompt": request.prompt,
            "results": results,
            "processing_time_ms": round(processing_time_ms, 2),
            "peak_memory_mb": round(mx.get_peak_memory() / (1024 * 1024), 2)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during segmentation: {str(e)}")


@app.post("/segment/box")
async def add_box_prompt(request: BoxPromptRequest):
    """Add a box prompt (positive or negative) and re-segment."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        state = session["state"]
        
        # Store prompted box for display
        if "prompted_boxes" not in state:
            state["prompted_boxes"] = []
        
        # Convert from normalized cxcywh to pixel xyxy for display
        img_w = state["original_width"]
        img_h = state["original_height"]
        cx, cy, w, h = request.box
        x_min = (cx - w / 2) * img_w
        y_min = (cy - h / 2) * img_h
        x_max = (cx + w / 2) * img_w
        y_max = (cy + h / 2) * img_h
        
        state["prompted_boxes"].append({
            "box": [x_min, y_min, x_max, y_max],
            "label": request.label
        })
        
        start_time = time.perf_counter()
        state = processor.add_geometric_prompt(request.box, request.label, state)
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        session["state"] = state
        
        return {
            "session_id": request.session_id,
            "box_type": "positive" if request.label else "negative",
            "results": serialize_state(state),
            "processing_time_ms": round(processing_time_ms, 2),
            "peak_memory_mb": round(mx.get_peak_memory() / (1024 * 1024), 2)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding box prompt: {str(e)}")


@app.post("/segment/point")
async def add_point_prompt(request: PointPromptRequest):
    """Add a point prompt (positive or negative) and re-segment."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        state = session["state"]
        
        # Store prompted point for display
        if "prompted_points" not in state:
            state["prompted_points"] = []
        
        # Convert from normalized to pixel coordinates for display
        img_w = state["original_width"]
        img_h = state["original_height"]
        x, y = request.point
        pixel_x = x * img_w
        pixel_y = y * img_h
        
        state["prompted_points"].append({
            "point": [pixel_x, pixel_y],
            "label": request.label
        })
        
        start_time = time.perf_counter()
        state = processor.add_point_prompt(request.point, request.label, state)
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        session["state"] = state
        
        return {
            "session_id": request.session_id,
            "point_type": "positive" if request.label else "negative",
            "results": serialize_state(state),
            "processing_time_ms": round(processing_time_ms, 2),
            "peak_memory_mb": round(mx.get_peak_memory() / (1024 * 1024), 2)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding point prompt: {str(e)}")


@app.post("/reset")
async def reset_prompts(request: SessionRequest):
    """Reset all prompts for a session."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        state = session["state"]
        
        start_time = time.perf_counter()
        processor.reset_all_prompts(state)
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        if "prompted_boxes" in state:
            del state["prompted_boxes"]
        
        if "prompted_points" in state:
            del state["prompted_points"]
        
        return {
            "session_id": request.session_id,
            "message": "All prompts reset",
            "results": serialize_state(state),
            "processing_time_ms": round(processing_time_ms, 2),
            "peak_memory_mb": round(mx.get_peak_memory() / (1024 * 1024), 2)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting prompts: {str(e)}")


@app.post("/confidence")
async def set_confidence(request: ConfidenceRequest):
    """Update confidence threshold (note: requires re-running inference)."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Update processor threshold
    processor.confidence_threshold = request.threshold
    
    return {
        "session_id": request.session_id,
        "threshold": request.threshold,
        "message": "Confidence threshold updated. Re-run segmentation to apply."
    }


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and free memory."""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": "Session deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


"""
FastAPI backend for SAM3 segmentation model.
Provides endpoints for image upload, text prompts, box prompts, and segmentation results.
Supports both normal images and DICOM volumes.
"""

import io
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import List

import mlx.core as mx
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
import pydicom

# Import DICOM utilities
from dicom_utils import (
    is_dicom_file,
    load_dicom_volume,
    mlx_slice_to_pil,
    get_default_window_for_modality,
    get_dicom_metadata
)

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


class SliceRequest(BaseModel):
    session_id: str
    slice_index: int


class PropagateRequest(BaseModel):
    session_id: str
    direction: str = "both"  # "forward", "backward", or "both"


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
async def upload_image(files: List[UploadFile] = File(...)):
    """Upload an image, DICOM file, or multiple DICOM files and initialize a session."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        # First, try to identify valid DICOM image files
        dicom_image_files = []
        regular_image_files = []
        
        for file in files:
            contents = await file.read()
            print(f"Processing: {file.filename}, size: {len(contents)}, content_type: {file.content_type}")
            
            # Check regular image files FIRST
            if file.content_type and file.content_type.startswith("image/"):
                # Regular image file
                regular_image_files.append((file, contents))
                print(f"  → Added as regular image (content-type)")
            
            elif file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')):
                # Has image extension
                regular_image_files.append((file, contents))
                print(f"  → Added as regular image (extension)")
            
            # Then check for DICOM
            elif is_dicom_file(contents):
                # Has DICM magic bytes - definitely DICOM
                print(f"  → Has DICM magic bytes")
                try:
                    dcm = pydicom.dcmread(io.BytesIO(contents), force=True)
                    if 'PixelData' in dcm or (0x7fe0, 0x0010) in dcm:
                        dicom_image_files.append((file, contents, dcm))
                        print(f"  → Added as DICOM image")
                    else:
                        print(f"  → Skipping: no pixel data")
                except Exception as e:
                    print(f"  → Skipping: {e}")
            
            else:
                # No standard image extension or magic bytes - might be DICOM without header
                print(f"  → Trying as DICOM without magic bytes")
                try:
                    dcm = pydicom.dcmread(io.BytesIO(contents), force=True)
                    if 'PixelData' in dcm or (0x7fe0, 0x0010) in dcm:
                        dicom_image_files.append((file, contents, dcm))
                        print(f"  → Added as DICOM image (no magic bytes)")
                    else:
                        print(f"  → Skipping: no pixel data")
                except Exception as e:
                    print(f"  → Skipping: {e}")
        
        print(f"Found {len(dicom_image_files)} DICOM images, {len(regular_image_files)} regular images")
        
        # Decide what to process based on what we found
        if len(dicom_image_files) > 0:
            # Process as DICOM volume
            if len(dicom_image_files) == 1:
                # Single DICOM file
                file, contents, dcm = dicom_image_files[0]
                print(f"Processing single DICOM file: {file.filename}")
                
                volume_mlx, dicom_metadata = load_dicom_volume(io.BytesIO(contents))
                
                # Ensure volume is at least 3D (single 2D slices get expanded)
                if volume_mlx.ndim == 2:
                    volume_mlx = mx.expand_dims(volume_mlx, 0)
                
                # Get first slice as PIL for SAM3
                first_slice_mlx = volume_mlx[0]
                total_slices = volume_mlx.shape[0]
                
                # Get default windowing for modality
                modality = dicom_metadata.get('modality', 'CT')
                window_center = dicom_metadata.get('window_center')
                window_width = dicom_metadata.get('window_width')
                
                if window_center is None or window_width is None:
                    window_center, window_width = get_default_window_for_modality(modality)
                
                # Convert to PIL
                image = mlx_slice_to_pil(
                    first_slice_mlx,
                    window_center=window_center,
                    window_width=window_width,
                    rescale_intercept=dicom_metadata.get('rescale_intercept', 0),
                    rescale_slope=dicom_metadata.get('rescale_slope', 1),
                    modality=modality
                )
                
                is_dicom = True
            
            else:
                # Multiple DICOM files - create volume
                print(f"Processing {len(dicom_image_files)} DICOM files as volume")
                
                # Sort by InstanceNumber or SliceLocation
                dicom_datasets = [dcm for _, _, dcm in dicom_image_files]
                dicom_datasets.sort(key=lambda x: (
                    getattr(x, 'InstanceNumber', 0),
                    getattr(x, 'SliceLocation', 0)
                ))
                
                # Extract pixel arrays and convert to MLX
                mlx_slices = []
                for dcm in dicom_datasets:
                    pixel_array = dcm.pixel_array
                    mlx_slices.append(mx.array(pixel_array))
                
                # Stack slices
                volume_mlx = mx.stack(mlx_slices, axis=0)
                total_slices = len(mlx_slices)
                
                # Get metadata from first slice
                dicom_metadata = get_dicom_metadata(dicom_datasets[0])
                
                # Get first slice as PIL for SAM3
                first_slice_mlx = volume_mlx[0]
                
                # Get default windowing for modality
                modality = dicom_metadata.get('modality', 'CT')
                window_center = dicom_metadata.get('window_center')
                window_width = dicom_metadata.get('window_width')
                
                if window_center is None or window_width is None:
                    window_center, window_width = get_default_window_for_modality(modality)
                
                # Convert to PIL
                image = mlx_slice_to_pil(
                    first_slice_mlx,
                    window_center=window_center,
                    window_width=window_width,
                    rescale_intercept=dicom_metadata.get('rescale_intercept', 0),
                    rescale_slope=dicom_metadata.get('rescale_slope', 1),
                    modality=modality
                )
                
                is_dicom = True
        
        elif len(regular_image_files) == 1:
            # Single regular image
            file, contents = regular_image_files[0]
            print(f"Processing single image file: {file.filename}")
            
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            
            # Convert to MLX and store as single-slice volume for consistency
            img_array = mx.array(np.array(image))
            volume_mlx = mx.expand_dims(img_array, 0)  # Add slice dimension
            
            total_slices = 1
            dicom_metadata = {}
            is_dicom = False
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"No valid image files found. Uploaded {len(files)} files but none contained valid image data."
            )
        
        # Create session
        session_id = str(uuid.uuid4())
        
        # Process first slice through model (timed)
        start_time = time.perf_counter()
        state = processor.set_image(image)
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Store session with volume data
        sessions[session_id] = {
            "state": state,
            "image_size": image.size,
            "volume_mlx": volume_mlx,
            "current_slice": 0,
            "total_slices": total_slices,
            "is_dicom": is_dicom,
            "dicom_metadata": dicom_metadata,
            "encoded_slice": 0,  # Track which slice is currently encoded
        }
        
        return {
            "session_id": session_id,
            "width": image.size[0],
            "height": image.size[1],
            "total_slices": total_slices,
            "current_slice": 0,
            "is_volume": total_slices > 1,
            "is_dicom": is_dicom,
            "modality": dicom_metadata.get('modality', 'unknown') if is_dicom else 'image',
            "message": f"{'DICOM volume' if is_dicom else 'Image'} uploaded and processed successfully",
            "processing_time_ms": round(processing_time_ms, 2),
            "peak_memory_mb": round(mx.get_peak_memory() / (1024 * 1024), 2)
        }
    
    except Exception as e:
        import traceback
        error_detail = f"Error processing file: {str(e)}"
        print(f"Upload error: {error_detail}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=error_detail)


@app.post("/segment/text")
async def segment_with_text(request: TextPromptRequest):
    """Segment image using text prompt."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        current_slice = session.get("current_slice", 0)
        encoded_slice = session.get("encoded_slice", -1)
        
        start_time = time.perf_counter()
        
        # Only encode if we haven't encoded this slice yet
        if encoded_slice != current_slice:
            # Get current slice
            volume_mlx = session["volume_mlx"]
            
            if volume_mlx.ndim == 3:
                slice_mlx = volume_mlx[current_slice]
            else:
                slice_mlx = volume_mlx
            
            # Get DICOM metadata
            dicom_metadata = session.get("dicom_metadata", {})
            
            # Convert to PIL
            image = mlx_slice_to_pil(
                slice_mlx,
                window_center=dicom_metadata.get('window_center'),
                window_width=dicom_metadata.get('window_width'),
                rescale_intercept=dicom_metadata.get('rescale_intercept', 0),
                rescale_slope=dicom_metadata.get('rescale_slope', 1),
                modality=dicom_metadata.get('modality', 'CT')
            )
            
            # Encode image through backbone
            state = processor.set_image(image)
            session["state"] = state
            session["image_size"] = image.size
            session["encoded_slice"] = current_slice
        else:
            # Use cached encoded state
            state = session["state"]
        
        # Apply text prompt to encoded state
        state = processor.set_text_prompt(request.prompt, state)
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        session["state"] = state
        
        results = serialize_state(state)
        
        return {
            "session_id": request.session_id,
            "prompt": request.prompt,
            "results": results,
            "processing_time_ms": round(processing_time_ms, 2),
            "peak_memory_mb": round(mx.get_peak_memory() / (1024 * 1024), 2)
        }
    
    except Exception as e:
        import traceback
        print(f"Error in text prompt: {e}")
        print(traceback.format_exc())
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
        current_slice = session.get("current_slice", 0)
        encoded_slice = session.get("encoded_slice", -1)
        
        start_time = time.perf_counter()
        
        # Only encode if we haven't encoded this slice yet
        if encoded_slice != current_slice:
            # Get current slice
            volume_mlx = session["volume_mlx"]
            
            if volume_mlx.ndim == 3:
                slice_mlx = volume_mlx[current_slice]
            else:
                slice_mlx = volume_mlx
            
            # Get DICOM metadata
            dicom_metadata = session.get("dicom_metadata", {})
            
            # Convert to PIL
            image = mlx_slice_to_pil(
                slice_mlx,
                window_center=dicom_metadata.get('window_center'),
                window_width=dicom_metadata.get('window_width'),
                rescale_intercept=dicom_metadata.get('rescale_intercept', 0),
                rescale_slope=dicom_metadata.get('rescale_slope', 1),
                modality=dicom_metadata.get('modality', 'CT')
            )
            
            # Encode image through backbone
            state = processor.set_image(image)
            session["state"] = state
            session["image_size"] = image.size
            session["encoded_slice"] = current_slice
        else:
            # Use cached encoded state
            state = session["state"]
        
        # Store prompted box for display
        if "prompted_boxes" not in state:
            state["prompted_boxes"] = []
        
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
        
        # Apply the geometric prompt to encoded state
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
        import traceback
        print(f"Error in box prompt: {e}")
        print(traceback.format_exc())
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
        current_slice = session.get("current_slice", 0)
        encoded_slice = session.get("encoded_slice", -1)
        
        start_time = time.perf_counter()
        
        # Only encode if we haven't encoded this slice yet
        if encoded_slice != current_slice:
            # Get current slice
            volume_mlx = session["volume_mlx"]
            
            if volume_mlx.ndim == 3:
                slice_mlx = volume_mlx[current_slice]
            else:
                slice_mlx = volume_mlx
            
            # Get DICOM metadata
            dicom_metadata = session.get("dicom_metadata", {})
            
            # Convert to PIL
            image = mlx_slice_to_pil(
                slice_mlx,
                window_center=dicom_metadata.get('window_center'),
                window_width=dicom_metadata.get('window_width'),
                rescale_intercept=dicom_metadata.get('rescale_intercept', 0),
                rescale_slope=dicom_metadata.get('rescale_slope', 1),
                modality=dicom_metadata.get('modality', 'CT')
            )
            
            # Encode image through backbone
            state = processor.set_image(image)
            session["state"] = state
            session["image_size"] = image.size
            session["encoded_slice"] = current_slice
        else:
            # Use cached encoded state
            state = session["state"]
        
        # Store prompted point for display
        if "prompted_points" not in state:
            state["prompted_points"] = []
        
        img_w = state["original_width"]
        img_h = state["original_height"]
        x, y = request.point
        pixel_x = x * img_w
        pixel_y = y * img_h
        
        state["prompted_points"].append({
            "point": [pixel_x, pixel_y],
            "label": request.label
        })
        
        # Apply the point prompt to encoded state
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
        import traceback
        print(f"Error in point prompt: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error adding point prompt: {str(e)}")


@app.post("/slice")
async def change_slice(request: SliceRequest):
    """Change to a different slice in the volume (fast - no SAM3 processing)."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Validate slice index
    total_slices = session.get("total_slices", 1)
    if request.slice_index < 0 or request.slice_index >= total_slices:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid slice index: {request.slice_index} (valid range: 0-{total_slices-1})"
        )
    
    try:
        # Just update the current slice index - don't process through SAM3 yet
        session["current_slice"] = request.slice_index
        
        # Get slice dimensions for response
        volume_mlx = session["volume_mlx"]
        if volume_mlx.ndim == 3:
            slice_mlx = volume_mlx[request.slice_index]
        else:
            slice_mlx = volume_mlx
        
        # Get dimensions (evaluate to get actual shape)
        mx.eval(slice_mlx)
        height, width = slice_mlx.shape[:2]
        
        # Clear segmentation results when changing slices
        # Mark encoded_slice as stale to force re-encoding on next prompt
        session["encoded_slice"] = -1
        session["state"] = {
            "original_width": width,
            "original_height": height,
        }
        
        return {
            "session_id": request.session_id,
            "slice_index": request.slice_index,
            "total_slices": total_slices,
            "width": width,
            "height": height,
            "results": {
                "original_width": width,
                "original_height": height,
            },
            "message": "Slice changed (segmentation will run on next prompt)",
            "processing_time_ms": 0,  # Instant!
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error changing slice: {str(e)}")


@app.get("/volume-info/{session_id}")
async def get_volume_info(session_id: str):
    """Get volume metadata (total slices, current slice, etc.)"""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "total_slices": session.get("total_slices", 1),
        "current_slice": session.get("current_slice", 0),
        "is_dicom": session.get("is_dicom", False),
        "is_volume": session.get("total_slices", 1) > 1,
        "modality": session.get("dicom_metadata", {}).get("modality", "unknown"),
        "image_size": session.get("image_size", (0, 0)),
    }


@app.get("/slice-image/{session_id}")
async def get_slice_image(session_id: str, slice_index: int = None):
    """Get the current slice as a PNG image for display."""
    from fastapi.responses import StreamingResponse
    
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Use current slice if not specified
        if slice_index is None:
            slice_index = session.get("current_slice", 0)
        
        # Get slice from MLX volume
        volume_mlx = session["volume_mlx"]
        if volume_mlx.ndim == 3:
            slice_mlx = volume_mlx[slice_index]
        else:
            slice_mlx = volume_mlx
        
        # Get DICOM metadata if available
        dicom_metadata = session.get("dicom_metadata", {})
        window_center = dicom_metadata.get('window_center')
        window_width = dicom_metadata.get('window_width')
        
        # Convert to PIL
        image = mlx_slice_to_pil(
            slice_mlx,
            window_center=window_center,
            window_width=window_width,
            rescale_intercept=dicom_metadata.get('rescale_intercept', 0),
            rescale_slope=dicom_metadata.get('rescale_slope', 1),
            modality=dicom_metadata.get('modality', 'CT')
        )
        
        # Convert to PNG bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(img_byte_arr, media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting slice image: {str(e)}")


@app.post("/propagate")
async def propagate_masks(request: PropagateRequest):
    """Propagate current slice segmentation to adjacent slices in the volume."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    total_slices = session.get("total_slices", 1)
    if total_slices <= 1:
        raise HTTPException(status_code=400, detail="Propagation requires a volume with multiple slices")
    
    state = session.get("state", {})
    if "masks" not in state:
        raise HTTPException(status_code=400, detail="No segmentation masks to propagate. Segment a slice first.")
    
    try:
        start_time = time.perf_counter()
        
        current_slice = session.get("current_slice", 0)
        volume_mlx = session["volume_mlx"]
        
        print(f"Propagating from slice {current_slice}, direction={request.direction}")
        
        volume_states = processor.propagate_to_volume(
            volume_mlx=volume_mlx,
            source_slice=current_slice,
            source_state=state,
            direction=request.direction,
        )
        
        # Store volume masks in session
        # Convert each state's masks to serializable format
        volume_masks = {}
        for slice_idx, slice_state in volume_states.items():
            volume_masks[slice_idx] = serialize_state(slice_state)
        
        session["volume_masks"] = volume_masks
        
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        propagated_slices = sorted(volume_states.keys())
        
        return {
            "session_id": request.session_id,
            "source_slice": current_slice,
            "propagated_slices": propagated_slices,
            "total_propagated": len(propagated_slices),
            "direction": request.direction,
            "processing_time_ms": round(processing_time_ms, 2),
            "peak_memory_mb": round(mx.get_peak_memory() / (1024 * 1024), 2),
        }
    
    except Exception as e:
        import traceback
        print(f"Error in propagation: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error during propagation: {str(e)}")


@app.get("/volume-masks/{session_id}/{slice_index}")
async def get_volume_mask(session_id: str, slice_index: int):
    """Get the propagated mask for a specific slice."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    volume_masks = session.get("volume_masks", {})
    
    # Try int key (from propagation) 
    mask_data = volume_masks.get(slice_index)
    if mask_data is None:
        raise HTTPException(status_code=404, detail=f"No mask for slice {slice_index}")
    
    return {
        "session_id": session_id,
        "slice_index": slice_index,
        "results": mask_data,
    }


@app.post("/reset")
async def reset_prompts(request: SessionRequest):
    """Reset all prompts for a session."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Clear prompts but keep cached backbone features for the current slice
        state = session.get("state", {})
        if "backbone_out" in state:
            processor.reset_all_prompts(state)
            # Also clear backend-specific prompt display data
            state.pop("prompted_boxes", None)
            state.pop("prompted_points", None)
        else:
            session["state"] = {}
        
        return {
            "session_id": request.session_id,
            "message": "All prompts reset",
            "results": {
                "original_width": state.get("original_width", 0),
                "original_height": state.get("original_height", 0),
            },
            "processing_time_ms": 0,
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


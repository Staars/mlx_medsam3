# Architecture Documentation

Technical architecture of MedSAM3 Studio.

## Overview

MedSAM3 Studio is a medical image segmentation system built on:
- **SAM3**: Meta's Segment Anything Model 3
- **MedSAM3**: Medical fine-tuning with LoRA
- **MLX**: Apple Silicon optimization
- **FastAPI + Next.js**: Modern web stack

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (Next.js)                    │
│  - Image upload & display                                │
│  - Interactive prompts (text/box/point)                  │
│  - Real-time mask visualization                          │
└────────────────┬────────────────────────────────────────┘
                 │ HTTP/REST
┌────────────────▼────────────────────────────────────────┐
│                 Backend (FastAPI)                        │
│  - Session management                                    │
│  - Image preprocessing                                   │
│  - Prompt handling                                       │
│  - Result serialization                                  │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│              Sam3Processor (MLX)                         │
│  - Image encoding                                        │
│  - Text/geometric prompt processing                      │
│  - Medical modality configuration                        │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│              Sam3Image Model (MLX)                       │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Vision-Language Backbone                        │   │
│  │  - ViT encoder (1024d, 32 layers)                │   │
│  │  - Text encoder (CLIP-style)                     │   │
│  │  - Feature fusion                                │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Transformer (DETR-style)                        │   │
│  │  - Encoder: 6 layers, cross-attention            │   │
│  │  - Decoder: 6 layers, 200 queries                │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Segmentation Head                               │   │
│  │  - Pixel decoder (FPN-style)                     │   │
│  │  - Mask prediction                               │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │  LoRA Layers (MedSAM3)                           │   │
│  │  - 916 LoRA tensors                              │   │
│  │  - Rank 16, Alpha 16.0                           │   │
│  │  - Applied to attention & MLP layers             │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Model Components

### 1. Vision Encoder (ViT)
- **Architecture**: Vision Transformer (Hiera-based)
- **Input**: 1008×1008 images
- **Patch size**: 14×14
- **Embedding**: 1024 dimensions
- **Layers**: 32 transformer blocks
- **Heads**: 16 attention heads
- **Features**: RoPE, global attention blocks

### 2. Text Encoder
- **Architecture**: CLIP-style transformer
- **Vocabulary**: 49,408 BPE tokens
- **Embedding**: 1024 dimensions
- **Layers**: 24 transformer blocks
- **Heads**: 16 attention heads
- **Output**: 256-d text features

### 3. Transformer (DETR)
- **Encoder**: 6 layers, cross-modal fusion
- **Decoder**: 6 layers, 200 object queries
- **Features**: Box refinement, DAC, presence tokens
- **Attention**: Multi-head (8 heads, 256-d)

### 4. Segmentation Head
- **Pixel Decoder**: 3-stage upsampling
- **Mask Prediction**: Per-query masks
- **Resolution**: Original image size
- **Output**: Binary masks + confidence scores

### 5. LoRA Fine-Tuning
- **Rank**: 16
- **Alpha**: 16.0 (scaling factor)
- **Dropout**: 0.0
- **Target Modules**: q_proj, v_proj, k_proj, out_proj, fc1, fc2
- **Components**: Vision encoder, text encoder, transformer
- **Parameters**: 18.5M trainable (vs 1B+ frozen)

## Data Flow

### Image Upload & Processing
```
1. User uploads image → Frontend
2. Frontend sends to /upload → Backend
3. Backend creates session
4. Image → Sam3Processor.set_image()
5. Vision encoder extracts features
6. Features cached in session state
7. Return session_id to frontend
```

### Text Prompt Segmentation
```
1. User types prompt → Frontend
2. Frontend sends to /segment/text → Backend
3. Backend retrieves session state
4. Text → Text encoder → text features
5. Vision + text features → Transformer
6. Transformer → object queries
7. Queries → Segmentation head → masks
8. Masks serialized (RLE) → Frontend
9. Frontend renders masks on canvas
```

### Point/Box Prompts
```
1. User clicks/draws → Frontend
2. Coordinates sent to /segment/point or /segment/box
3. Geometric encoder processes coordinates
4. Combined with cached image features
5. Transformer refines segmentation
6. Updated masks → Frontend
```

## Medical Modality System

### Configuration
Each modality has:
- **Window settings**: Intensity normalization (CT: [-160, 240])
- **Preprocessing**: Modality-specific transforms
- **Prompt suggestions**: Common anatomical terms
- **Confidence thresholds**: Adjusted per modality

### Supported Modalities
```python
MODALITIES = {
    "ct": {"window": [-160, 240], "normalize": "hounsfield"},
    "mri": {"window": [0, 4000], "normalize": "percentile"},
    "xray": {"window": [0, 255], "normalize": "minmax"},
    "ultrasound": {"window": [0, 255], "normalize": "minmax"},
    "microscopy": {"window": [0, 255], "normalize": "standard"},
    # ... more modalities
}
```

## LoRA Weight Loading

### Process
```python
1. Load base SAM3 model (frozen weights)
2. Inject LoRA layers into target modules
3. Load MedSAM3 LoRA weights from safetensors
4. Remap keys (MedSAM3 → MLX naming)
5. Apply LoRA weights to injected layers
6. Evaluate model (compile for inference)
```

### Key Remapping
```python
KEY_MAPPINGS = {
    "backbone.vision_backbone": "backbone.visual",
    "backbone.language_backbone": "backbone.text",
    "geometry_encoder": "input_geometry_encoder",
}
```

## API Endpoints

### Core Endpoints
- `POST /upload` - Upload image, create session
- `POST /segment/text` - Text prompt segmentation
- `POST /segment/box` - Box prompt refinement
- `POST /segment/point` - Point prompt refinement
- `POST /reset` - Clear all prompts
- `POST /modality` - Set medical modality
- `GET /modalities` - List available modalities
- `GET /health` - Health check

### Response Format
```json
{
  "session_id": "uuid",
  "results": {
    "masks": [{"counts": [0, 100, 50, ...], "size": [H, W]}],
    "boxes": [[x1, y1, x2, y2], ...],
    "scores": [0.95, 0.87, ...],
    "original_width": 1024,
    "original_height": 768
  },
  "processing_time_ms": 150.23,
  "peak_memory_mb": 2048.5
}
```

## Performance Optimization

### MLX Optimizations
- **Unified memory**: Shared CPU/GPU memory
- **Lazy evaluation**: Compute only when needed
- **Graph compilation**: Optimized execution
- **Metal acceleration**: Native GPU kernels

### Caching Strategy
- Image features cached per session
- Text embeddings cached per prompt
- Geometric features recomputed (lightweight)

### Memory Management
- Session cleanup on disconnect
- Periodic garbage collection
- Peak memory monitoring

## File Structure

```
mlx_medsam3/
├── sam3/                      # Core model
│   ├── model/                 # Model components
│   │   ├── sam3_image.py     # Main model class
│   │   ├── encoder.py        # Vision encoder
│   │   ├── decoder.py        # Transformer decoder
│   │   ├── text_encoder_ve.py # Text encoder
│   │   └── ...
│   ├── model_builder.py       # Model construction
│   ├── lora.py               # LoRA implementation
│   └── medical_utils.py      # Medical utilities
├── app/
│   ├── backend/
│   │   └── main.py           # FastAPI server
│   └── frontend/             # Next.js app
├── scripts/
│   ├── download_medsam3_weights.py
│   ├── convert_medsam3_to_mlx.py
│   └── verify_mlx_weights.py
└── examples/                  # Usage examples
```

## Development

### Adding New Modalities
1. Add config to `medical_utils.py`
2. Define window settings and normalization
3. Add prompt suggestions
4. Test with sample images

### Custom LoRA Training
1. Prepare medical dataset (COCO format)
2. Configure LoRA parameters
3. Use `examples/lora_training_example.py`
4. Convert trained weights to MLX
5. Load in model builder

### Extending Prompts
- Text: Modify text encoder input
- Geometric: Update geometry encoder
- New types: Add to processor interface

## Testing

### Unit Tests
```bash
uv run pytest tests/
```

### Integration Tests
```bash
# Test backend
curl http://localhost:8000/health

# Test full pipeline
uv run python examples/medical_inference_example.py
```

### Performance Benchmarks
```bash
uv run python examples/benchmark.py
```

## Deployment

### Docker (TODO)
```dockerfile
FROM python:3.13-slim
# ... setup
```

### Cloud Deployment
- Backend: FastAPI on cloud GPU
- Frontend: Vercel/Netlify
- Weights: S3/GCS storage

## References

- [SAM3 Paper](https://ai.meta.com/sam3)
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MedSAM3 Weights](https://huggingface.co/lal-Joey/MedSAM3_v1)
- [DETR Paper](https://arxiv.org/abs/2005.12872)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

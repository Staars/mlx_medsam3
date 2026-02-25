# MLX MedSAM3

Medical image segmentation powered by SAM3 and MedSAM3 fine-tuned weights, optimized for Apple Silicon with MLX.

> **Note**: This is a fork of [mlx_sam3](https://github.com/Deekshith-Dade/mlx_sam3) by Deekshith-Dade, enhanced with MedSAM3 medical fine-tuning, point prompts, and medical modality support.

## What's New in This Fork

This fork adds medical imaging capabilities to the original mlx_sam3:

- 🏥 **MedSAM3 Integration**: Fine-tuned LoRA weights for 330+ medical concepts
- 🎯 **Point Prompts**: Click-based segmentation refinement
- 🔬 **Medical Modalities**: Support for CT, MRI, X-ray, Ultrasound, Microscopy, and more
- ⚙️ **Automated Setup**: One-command installation with weight download/conversion
- 📊 **Medical Utilities**: Modality-specific preprocessing and prompt suggestions
- 🎨 **Enhanced UI**: Medical-focused web interface

![MedSAM3 Studio](https://img.shields.io/badge/MLX-Optimized-blue) ![Python](https://img.shields.io/badge/Python-3.13+-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

- 🏥 **Medical-Specific**: Fine-tuned on 330+ medical concepts across 10+ imaging modalities
- 🚀 **Apple Silicon Optimized**: Native MLX implementation for M1/M2/M3 chips
- 🎯 **Multiple Prompt Types**: Text, box, and point prompts for interactive segmentation
- 🖥️ **Web Interface**: Modern Next.js frontend with real-time segmentation
- 📊 **Medical Modalities**: CT, MRI, X-ray, Ultrasound, Microscopy, and more

## Quick Start

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.13+
- Node.js 18+ (for frontend)
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/Staars/mlx_medsam3.git
cd mlx_medsam3

# Install dependencies
uv sync

# Download and convert MedSAM3 weights
uv run python scripts/download_medsam3_weights.py
uv run python scripts/convert_medsam3_to_mlx.py

# Verify installation
uv run python scripts/verify_mlx_weights.py
```

### Running the App

```bash
# Start both backend and frontend
cd app
./run.sh

# Or start separately:
# Backend: uv run python app/backend/main.py
# Frontend: cd app/frontend && npm run dev
```

Open http://localhost:3000 in your browser.

## Usage

### Web Interface

1. **Upload Image**: Drag & drop or click to upload medical images
2. **Set Modality**: Choose imaging type (CT, MRI, X-ray, etc.)
3. **Segment**: Use text prompts ("liver", "tumor") or draw boxes/points
4. **Refine**: Add positive/negative prompts to refine segmentation
5. **Export**: Download masks in various formats

### Python API

```python
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image

# Load model with MedSAM3 weights
model = build_sam3_image_model()
processor = Sam3Processor(model)

# Set medical modality
processor.set_modality("ct")

# Process image
image = Image.open("ct_scan.jpg")
state = processor.set_image(image)

# Segment with text prompt
state = processor.set_text_prompt("liver", state)

# Access results
masks = state["masks"]
boxes = state["boxes"]
scores = state["scores"]
```

### Supported Medical Modalities

| Modality | Description | Example Prompts |
|----------|-------------|-----------------|
| CT | Computed Tomography | liver, kidney, tumor, lesion |
| MRI | Magnetic Resonance | brain, ventricle, white matter |
| X-ray | Radiography | lung, heart, rib, fracture |
| Ultrasound | Sonography | fetus, gallbladder, kidney stone |
| Microscopy | Pathology | cell, nucleus, mitochondria |
| Endoscopy | Internal imaging | polyp, ulcer, bleeding |
| Fundus | Retinal imaging | optic disc, blood vessel, macula |
| Dermoscopy | Skin imaging | lesion, melanoma, nevus |
| Mammography | Breast imaging | mass, calcification, density |
| OCT | Optical Coherence | retinal layer, fluid, drusen |

## Project Structure

```
mlx_medsam3/
├── sam3/                  # Core MLX model implementation
│   ├── model/            # Model architecture
│   ├── lora.py           # LoRA fine-tuning support
│   └── medical_utils.py  # Medical-specific utilities
├── app/                   # Web application
│   ├── backend/          # FastAPI backend
│   └── frontend/         # Next.js frontend
├── scripts/              # Setup and utility scripts
│   ├── download_medsam3_weights.py
│   ├── convert_medsam3_to_mlx.py
│   └── verify_mlx_weights.py
├── examples/             # Usage examples
│   ├── medical_inference_example.py
│   ├── point_prompt_example.py
│   └── lora_training_example.py
└── configs/              # Configuration files
    └── medical_presets.yaml
```

## Advanced Features

### Point Prompts

```python
# Add positive point (foreground)
state = processor.add_point_prompt([0.5, 0.5], label=True, state=state)

# Add negative point (background)
state = processor.add_point_prompt([0.2, 0.2], label=False, state=state)
```

### Box Prompts

```python
# Box format: [center_x, center_y, width, height] (normalized 0-1)
state = processor.add_geometric_prompt([0.5, 0.5, 0.3, 0.3], label=True, state=state)
```

### LoRA Fine-Tuning

See `examples/lora_training_example.py` for custom fine-tuning on your medical datasets.

## Performance

On Apple M2 Max:
- Image encoding: ~200ms
- Text prompt segmentation: ~150ms
- Point/box refinement: ~50ms
- Peak memory: ~2GB

## Model Weights

The project uses two sets of weights:

1. **Base SAM3**: Downloaded automatically from HuggingFace
2. **MedSAM3 LoRA**: Fine-tuned medical weights (71MB)
   - Source: `lal-Joey/MedSAM3_v1`
   - 916 LoRA tensors
   - 18.5M parameters

## API Documentation

When running the backend, visit http://localhost:8000/docs for interactive API documentation.

### Key Endpoints

- `POST /upload` - Upload and process image
- `POST /modality` - Set medical imaging modality
- `POST /segment/text` - Segment with text prompt
- `POST /segment/box` - Add box prompt
- `POST /segment/point` - Add point prompt
- `POST /reset` - Reset all prompts
- `GET /modalities` - List available modalities

## Development

### Setup Development Environment

```bash
# Install dev dependencies
uv sync --dev

# Run tests
uv run pytest

# Format code
uv run black .
uv run isort .
```

### Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed technical documentation.

## Troubleshooting

### Backend won't start
- Verify weights exist: `ls -lh weights/`
- Check logs for errors
- Try base model: `rm weights/medsam3_lora.safetensors`

### Poor segmentation quality
- Verify MedSAM3 weights loaded (check startup logs)
- Set correct medical modality
- Use medical-specific prompts
- Try multiple prompts for refinement

### Memory issues
- Reduce image size before upload
- Close other applications
- Use base SAM3 without LoRA

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request to https://github.com/Staars/mlx_medsam3

## Citation

If you use this work, please cite both the original mlx_sam3 and this medical enhancement:

```bibtex
@software{mlx_medsam3,
  title={MLX MedSAM3: Medical Image Segmentation with SAM3},
  author={Staars},
  year={2025},
  url={https://github.com/Staars/mlx_medsam3},
  note={Fork of mlx_sam3 by Deekshith-Dade with MedSAM3 medical fine-tuning}
}

@software{mlx_sam3,
  title={MLX SAM3: SAM3 Implementation for Apple Silicon},
  author={Deekshith-Dade},
  year={2024},
  url={https://github.com/Deekshith-Dade/mlx_sam3}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [Deekshith-Dade](https://github.com/Deekshith-Dade) for the original [mlx_sam3](https://github.com/Deekshith-Dade/mlx_sam3) implementation
- Meta AI for SAM3 architecture
- MedSAM3 team for medical fine-tuning
- MLX team for Apple Silicon optimization
- HuggingFace for model hosting

## Links

- **Original Project**: [mlx_sam3 by Deekshith-Dade](https://github.com/Deekshith-Dade/mlx_sam3)
- [SAM3 Paper](https://ai.meta.com/sam3)
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MedSAM3 Weights](https://huggingface.co/lal-Joey/MedSAM3_v1)
- [Report Issues](https://github.com/Staars/mlx_medsam3/issues)

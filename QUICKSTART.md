# Quick Start Guide

Get MedSAM3 Studio running in 5 minutes.

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.13+
- Node.js 18+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed

## Step 1: Clone & Install (2 min)

```bash
git clone https://github.com/Staars/mlx_medsam3.git
cd mlx_medsam3
uv sync
```

## Step 2: Download Weights (2 min)

```bash
# Download MedSAM3 weights from HuggingFace
uv run python scripts/download_medsam3_weights.py

# Convert to MLX format
uv run python scripts/convert_medsam3_to_mlx.py

# Verify (optional)
uv run python scripts/verify_mlx_weights.py
```

Expected output:
```
✓ Downloaded 1 files (71 MB)
✓ Converted 916 LoRA tensors (70.68 MB)
✓ Verification complete!
```

## Step 3: Run the App (1 min)

```bash
cd app
./run.sh
```

This starts:
- Backend: http://localhost:8000
- Frontend: http://localhost:3000

## Step 4: Test It!

1. Open http://localhost:3000
2. Upload a medical image (CT, MRI, X-ray, etc.)
3. Select modality (e.g., "CT")
4. Type a prompt (e.g., "liver")
5. See the segmentation!

## Quick Examples

### Example 1: CT Scan Liver Segmentation

```python
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image

model = build_sam3_image_model()
processor = Sam3Processor(model)
processor.set_modality("ct")

image = Image.open("ct_scan.jpg")
state = processor.set_image(image)
state = processor.set_text_prompt("liver", state)

# Get results
masks = state["masks"]
```

### Example 2: Point Prompts

```python
# Add positive point at center
state = processor.add_point_prompt([0.5, 0.5], label=True, state=state)

# Add negative point to exclude region
state = processor.add_point_prompt([0.2, 0.2], label=False, state=state)
```

### Example 3: Box Prompts

```python
# Box: [center_x, center_y, width, height] (normalized 0-1)
state = processor.add_geometric_prompt([0.5, 0.5, 0.3, 0.3], label=True, state=state)
```

## Troubleshooting

### "Module not found" errors
```bash
uv sync  # Re-sync dependencies
```

### Backend won't start
```bash
# Check if weights exist
ls -lh weights/

# If missing, re-download
uv run python scripts/download_medsam3_weights.py
uv run python scripts/convert_medsam3_to_mlx.py
```

### Frontend won't start
```bash
cd app/frontend
npm install
npm run dev
```

### Poor segmentation
- Set correct modality for your image type
- Use medical-specific terms ("tumor" not "growth")
- Try multiple prompts to refine

## Next Steps

- Read [README.md](README.md) for full documentation
- Check [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
- Explore `examples/` for more use cases
- Visit http://localhost:8000/docs for API documentation

## Performance Tips

- Use images < 2000px for faster processing
- Close other apps to free memory
- First run is slower (model loading)
- Subsequent runs are much faster

## Getting Help

- Check logs in terminal for errors
- Visit http://localhost:8000/health to test backend
- Open browser console (F12) for frontend errors
- Report issues on GitHub

Enjoy using MedSAM3 Studio! 🏥

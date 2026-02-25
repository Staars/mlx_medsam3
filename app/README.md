# MLX MedSAM3 - Web Application

An interactive web application for medical image segmentation using the MedSAM3 model. Features a modern Next.js frontend with a FastAPI backend, supporting 10+ medical imaging modalities and LoRA fine-tuning.

## Features

- **Medical Modalities**: Support for CT, MRI, X-Ray, Ultrasound, Microscopy, Endoscopy, and more
- **Text Prompts**: Describe medical concepts (e.g., "lung nodule", "tumor", "cell")
- **Box Prompts**: Draw bounding boxes to include or exclude regions
- **Point Prompts**: Click points to refine segmentation
- **Modality-Specific Thresholds**: Automatic configuration per imaging type
- **Prompt Suggestions**: Get relevant medical concepts for each modality
- **Real-time Visualization**: See segmentation masks and bounding boxes overlaid on your image
- **Session Management**: Multiple users can use the app simultaneously
- **LoRA Support**: Parameter-efficient fine-tuning for medical domain adaptation

## Architecture

```
app/
├── backend/           # FastAPI server
│   ├── main.py       # API endpoints
│   └── requirements.txt
└── frontend/          # Next.js app
    └── src/
        ├── app/       # Pages and layout
        ├── components/  # React components
        └── lib/       # API client and utilities
```

## Prerequisites

- Python 3.10+ with the SAM3 model dependencies installed
- Node.js 18+
- SAM3 model weights at `sam3-mod-weights/model.safetensors`

## Quick Start

Run both servers with a single command:

```bash
cd app
./run.sh
```

This will start:
- **Backend**: http://localhost:8000
- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs

Press `Ctrl+C` to stop both servers.

---

## Manual Setup

### Backend

1. Install Python dependencies:

```bash
cd app/backend
pip install -r requirements.txt
```

2. Start the backend server:

```bash
python main.py
# Or with uvicorn directly:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The backend will load the SAM3 model on startup (this may take a minute).

### Frontend

1. Install Node dependencies:

```bash
cd app/frontend
npm install
```

2. Start the development server:

```bash
npm run dev
```

3. Open [http://localhost:3000](http://localhost:3000) in your browser

## Usage

1. **Upload an Image**: Click the upload area or drag & drop a medical image
2. **Select Modality**: Choose the appropriate medical imaging modality (CT, MRI, X-Ray, etc.)
3. **View Suggestions**: Click on suggested medical concepts or enter your own
4. **Text Segmentation**: Enter a medical concept like "lung nodule" or "tumor" and press Enter
5. **Box Prompts**: 
   - Select "Include" to draw boxes around regions you want to segment
   - Select "Exclude" to draw boxes around regions to exclude
   - Draw by clicking and dragging on the image
6. **Point Prompts**:
   - Select "Include" or "Exclude" mode
   - Click points on the image to refine segmentation
7. **Reset**: Click "Clear All Prompts" to start fresh with the same image

## Medical Modalities

The app supports the following medical imaging modalities with optimized thresholds:

- **CT** (Computed Tomography) - confidence: 0.6
- **MRI** (Magnetic Resonance Imaging) - confidence: 0.6
- **X-Ray** - confidence: 0.55
- **Ultrasound** - confidence: 0.45
- **Microscopy** - confidence: 0.7
- **Endoscopy** - confidence: 0.5
- **Histopathology** - confidence: 0.65
- **Dermoscopy** - confidence: 0.6
- **OCT** (Optical Coherence Tomography) - confidence: 0.6
- **General** - confidence: 0.5

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check backend status |
| `/upload` | POST | Upload image and create session |
| `/segment/text` | POST | Segment with text prompt |
| `/segment/box` | POST | Add box prompt |
| `/segment/point` | POST | Add point prompt |
| `/modalities` | GET | List available medical modalities |
| `/medical/suggestions/{modality}` | GET | Get prompt suggestions for modality |
| `/modality` | POST | Set medical modality for session |
| `/reset` | POST | Reset all prompts |
| `/session/{id}` | DELETE | Delete session |

## Environment Variables

### Frontend

- `NEXT_PUBLIC_API_URL`: Backend API URL (default: `http://localhost:8000`)

## Development

### Frontend

```bash
cd app/frontend
npm run dev      # Development server
npm run build    # Production build
npm run lint     # Lint code
```

### Backend

```bash
cd app/backend
uvicorn main:app --reload  # Hot reload during development
```

## Tech Stack

- **Frontend**: Next.js 15, React 19, Tailwind CSS 4, TypeScript
- **Backend**: FastAPI, Uvicorn
- **Model**: MedSAM3 (MLX implementation with LoRA support)
- **Medical Imaging**: 10+ modality presets with 330+ medical concepts

## Documentation

- **Medical Features**: See [MEDICAL_FRONTEND_GUIDE.md](MEDICAL_FRONTEND_GUIDE.md)
- **Full Documentation**: See [../MEDSAM3_ENHANCEMENTS.md](../MEDSAM3_ENHANCEMENTS.md)
- **Quick Start**: See [../MEDICAL_QUICKSTART.md](../MEDICAL_QUICKSTART.md)

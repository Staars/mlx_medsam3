# MedSAM3 Integration Scripts

This directory contains scripts for setting up MedSAM3 LoRA weights in the MLX SAM3 implementation.

## 📁 Scripts Overview

### 1. `setup_medsam3.sh` - Complete Automated Setup
**Purpose:** One-command setup for MedSAM3 integration

**Usage:**
```bash
./setup_medsam3.sh
```

**What it does:**
- Checks prerequisites
- Downloads MedSAM3 weights from HuggingFace
- Converts PyTorch weights to MLX format
- Verifies installation
- Creates necessary directories

**Time:** ~15 minutes (mostly download time)

---

### 2. `download_medsam3_weights.py` - Download Weights
**Purpose:** Download MedSAM3 LoRA weights from HuggingFace

**Usage:**
```bash
python3 download_medsam3_weights.py
```

**Requirements:**
- HuggingFace account
- `huggingface_hub` package
- Internet connection

**Output:**
- Downloads to: `../weights/medsam3/`
- Files: `*.pt`, `*.pth`, `*.bin`, `*.safetensors`, `*.json`

**Troubleshooting:**
```bash
# If authentication fails
huggingface-cli login

# If repo not found
# Check: https://huggingface.co/lal-Joey/MedSAM3_v1
```

---

### 3. `convert_medsam3_to_mlx.py` - Convert Weights
**Purpose:** Convert PyTorch LoRA weights to MLX format

**Usage:**
```bash
python3 convert_medsam3_to_mlx.py
```

**Requirements:**
- PyTorch installed
- MLX installed
- Downloaded PyTorch weights

**Input:**
- `../weights/medsam3/*.pt` (PyTorch weights)

**Output:**
- `../weights/medsam3_lora.safetensors` (MLX weights)

**What it does:**
1. Finds PyTorch weight file
2. Loads PyTorch tensors
3. Converts to numpy arrays
4. Converts to MLX arrays
5. Saves as MLX safetensors

**Troubleshooting:**
```bash
# If PyTorch not found
pip install torch

# If input file not found
ls ../weights/medsam3/
# Run download script first
```

---

### 4. `verify_mlx_weights.py` - Verify Installation
**Purpose:** Verify MLX weights are valid and properly structured

**Usage:**
```bash
python3 verify_mlx_weights.py
```

**Requirements:**
- MLX installed
- Converted MLX weights

**What it checks:**
- File exists
- Can be loaded
- Contains LoRA keys
- Structure is correct
- Tensor shapes valid

**Expected Output:**
```
✓ Loaded X tensors
LoRA-related tensors: X
LoRA A matrices: X
LoRA B matrices: X
✓ Weights appear to be valid LoRA weights
```

**Troubleshooting:**
```bash
# If file not found
ls ../weights/medsam3_lora.safetensors
# Run conversion script first

# If no LoRA keys found
# Check conversion script output
# Verify input PyTorch file is LoRA weights
```

---

## 🚀 Quick Start

### Option 1: Automated (Recommended)
```bash
./setup_medsam3.sh
```

### Option 2: Manual Steps
```bash
# 1. Download
python3 download_medsam3_weights.py

# 2. Convert
python3 convert_medsam3_to_mlx.py

# 3. Verify
python3 verify_mlx_weights.py
```

---

## 📋 Prerequisites

**Required Packages:**
```bash
pip install huggingface_hub torch mlx
```

**HuggingFace Authentication:**
```bash
huggingface-cli login
# Enter your token when prompted
```

**Disk Space:**
- ~1GB for PyTorch weights
- ~1GB for MLX weights
- Total: ~2GB

---

## 🔍 Verification

After running scripts, verify:

```bash
# Check files exist
ls -lh ../weights/medsam3/
ls -lh ../weights/medsam3_lora.safetensors

# Check file sizes
# PyTorch: ~500MB-1GB
# MLX: ~500MB-1GB

# Verify structure
python3 verify_mlx_weights.py
```

---

## 🐛 Troubleshooting

### Common Issues

**1. "Repository not found"**
- Solution: Login to HuggingFace
- Command: `huggingface-cli login`

**2. "No module named 'torch'"**
- Solution: Install PyTorch
- Command: `pip install torch`

**3. "Weight file not found"**
- Solution: Run download script first
- Command: `python3 download_medsam3_weights.py`

**4. "Conversion failed"**
- Check PyTorch installation
- Verify input file exists
- Check disk space

**5. "No LoRA keys found"**
- Verify input file is LoRA weights
- Check HuggingFace repo for correct file
- May need different file from repo

---

## 📊 Expected Results

### After Download
```
weights/medsam3/
├── best_lora_weights.pt  (or similar)
├── config.json
└── README.md
```

### After Conversion
```
weights/
├── medsam3/
│   └── *.pt
└── medsam3_lora.safetensors  ← MLX weights
```

### After Verification
```
✓ Loaded 500+ tensors
LoRA-related tensors: 200+
✓ Weights appear to be valid LoRA weights
```

---

## 🔄 Workflow

```
1. Download (HuggingFace) → PyTorch weights
                              ↓
2. Convert (PyTorch→MLX)   → MLX weights
                              ↓
3. Verify (Check structure) → Validation
                              ↓
4. Backend loads weights    → Ready!
```

---

## 📚 Documentation

- **Quick Start:** `../MEDSAM3_QUICKSTART.md`
- **Full Plan:** `../MEDSAM3_INTEGRATION_PLAN.md`
- **Checklist:** `../MEDSAM3_CHECKLIST.md`
- **Summary:** `../MEDSAM3_SOLUTION_SUMMARY.md`

---

## 🆘 Support

If scripts fail:
1. Check prerequisites installed
2. Verify HuggingFace authentication
3. Check internet connection
4. Review error messages
5. Consult troubleshooting section

---

## 📝 Notes

- Scripts are idempotent (safe to re-run)
- Download is one-time (cached)
- Conversion overwrites existing MLX weights
- Verification is non-destructive

---

## ✅ Success Criteria

Scripts successful when:
- ✅ No error messages
- ✅ All files created
- ✅ Verification passes
- ✅ File sizes reasonable (~500MB-1GB each)

---

**Need help? See the full documentation in the parent directory.**

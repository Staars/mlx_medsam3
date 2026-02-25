#!/usr/bin/env python3
"""Download MedSAM3 LoRA weights from HuggingFace."""

from huggingface_hub import hf_hub_download, list_repo_files
from pathlib import Path
import sys

MEDSAM3_REPO = "lal-Joey/MedSAM3_v1"
WEIGHTS_DIR = Path(__file__).parent.parent / "weights" / "medsam3"

def download_medsam3():
    """Download MedSAM3 weights from HuggingFace."""
    print("=" * 70)
    print(f"Downloading MedSAM3 weights from {MEDSAM3_REPO}")
    print("=" * 70)
    
    try:
        # Create weights directory
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {WEIGHTS_DIR}")
        
        # List files in repo
        print(f"\nListing files in repository...")
        files = list_repo_files(MEDSAM3_REPO)
        print(f"✓ Found {len(files)} files in repository")
        
        # Filter for weight files
        weight_files = [f for f in files if f.endswith(('.pt', '.pth', '.bin', '.safetensors', '.json'))]
        print(f"\nWeight files to download: {len(weight_files)}")
        for f in weight_files:
            print(f"  - {f}")
        
        # Download weight files
        downloaded = []
        for filename in weight_files:
            try:
                print(f"\n⬇️  Downloading {filename}...")
                filepath = hf_hub_download(
                    repo_id=MEDSAM3_REPO,
                    filename=filename,
                    local_dir=str(WEIGHTS_DIR),
                )
                downloaded.append(filepath)
                print(f"✓ Saved to {filepath}")
            except Exception as e:
                print(f"⚠️  Could not download {filename}: {e}")
        
        print("\n" + "=" * 70)
        print(f"✓ Downloaded {len(downloaded)} files")
        print("=" * 70)
        
        return WEIGHTS_DIR
        
    except Exception as e:
        print(f"\n❌ Error downloading weights: {e}")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Verify HuggingFace authentication:")
        print("   huggingface-cli login")
        print("3. Check repository exists: https://huggingface.co/lal-Joey/MedSAM3_v1")
        sys.exit(1)

if __name__ == "__main__":
    download_medsam3()

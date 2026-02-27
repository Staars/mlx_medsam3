"""
DICOM utilities with MLX-first approach.
Minimal NumPy usage - only for pydicom compatibility at I/O boundaries.
"""

import io
import mlx.core as mx
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, List, Tuple, Optional
import pydicom
from pydicom.errors import InvalidDicomError


def is_dicom_file(file_data: Union[bytes, io.BytesIO, str, Path]) -> bool:
    """
    Detect DICOM file by magic bytes (doesn't rely on extension).
    Checks for 'DICM' at byte offset 128.
    
    Args:
        file_data: Bytes, BytesIO, or file path
        
    Returns:
        True if DICOM file, False otherwise
    """
    try:
        # Check magic bytes only - don't use pydicom fallback
        if isinstance(file_data, (str, Path)):
            with open(file_data, 'rb') as f:
                f.seek(128)
                magic = f.read(4)
        elif isinstance(file_data, bytes):
            magic = file_data[128:132] if len(file_data) >= 132 else b''
        elif isinstance(file_data, io.BytesIO):
            pos = file_data.tell()
            file_data.seek(128)
            magic = file_data.read(4)
            file_data.seek(pos)  # Reset position
        else:
            return False
        
        return magic == b'DICM'
            
    except Exception:
        return False


def load_dicom_volume(path: Union[str, Path, io.BytesIO, bytes]) -> Tuple[mx.array, dict]:
    """
    Load DICOM file(s) and return as MLX array with metadata.
    
    Args:
        path: Single DICOM file, directory containing DICOM series, BytesIO, or bytes
        
    Returns:
        Tuple of (mx.array, metadata_dict)
        - mx.array of shape (slices, height, width) or (height, width) for single slice
        - metadata dict with DICOM tags
    """
    slices = []
    metadata = {}
    
    # Handle different input types
    if isinstance(path, (str, Path)):
        path = Path(path)
        
        if path.is_dir():
            # Load all DICOM files from directory
            dicom_files = []
            for file_path in path.iterdir():
                if file_path.is_file() and is_dicom_file(file_path):
                    dicom_files.append(file_path)
            
            if not dicom_files:
                raise ValueError(f"No DICOM files found in directory: {path}")
            
            # Load and sort by instance number
            dicom_datasets = []
            for file_path in dicom_files:
                try:
                    dcm = pydicom.dcmread(file_path, force=True)
                    # Only add if it has pixel data
                    if 'PixelData' in dcm or (0x7fe0, 0x0010) in dcm:
                        dicom_datasets.append(dcm)
                    else:
                        print(f"Warning: {file_path.name} has no pixel data element")
                except Exception as e:
                    print(f"Warning: Failed to read {file_path}: {e}")
            
            # Sort by InstanceNumber or SliceLocation
            dicom_datasets.sort(key=lambda x: (
                getattr(x, 'InstanceNumber', 0),
                getattr(x, 'SliceLocation', 0)
            ))
            
            # Extract pixel arrays
            for dcm in dicom_datasets:
                pixel_array = dcm.pixel_array
                slices.append(pixel_array)
            
            # Get metadata from first slice
            if dicom_datasets:
                metadata = get_dicom_metadata(dicom_datasets[0])
        
        else:
            # Single DICOM file
            dcm = pydicom.dcmread(path, force=True)
            # Check if it has pixel data tags
            if 'PixelData' in dcm or (0x7fe0, 0x0010) in dcm:
                slices.append(dcm.pixel_array)
                metadata = get_dicom_metadata(dcm)
            else:
                raise ValueError("DICOM file has no pixel data element")
    
    elif isinstance(path, (io.BytesIO, bytes)):
        # Load from bytes
        try:
            dcm = pydicom.dcmread(path, force=True)
            # Check if it has pixel data tags
            if 'PixelData' in dcm or (0x7fe0, 0x0010) in dcm:
                slices.append(dcm.pixel_array)
                metadata = get_dicom_metadata(dcm)
            else:
                raise ValueError("DICOM file has no pixel data element")
        except InvalidDicomError as e:
            raise ValueError(f"Invalid DICOM data: {e}")
    
    else:
        raise ValueError(f"Unsupported input type: {type(path)}")
    
    if not slices:
        raise ValueError("No DICOM slices loaded")
    
    # Convert to MLX arrays immediately (minimize NumPy usage)
    mlx_slices = [mx.array(slice_np) for slice_np in slices]
    
    # Stack if multiple slices
    if len(mlx_slices) > 1:
        volume_mlx = mx.stack(mlx_slices, axis=0)
    else:
        volume_mlx = mlx_slices[0]
    
    # Evaluate to materialize the array
    mx.eval(volume_mlx)
    
    return volume_mlx, metadata


def get_dicom_metadata(dcm: pydicom.Dataset) -> dict:
    """
    Extract relevant DICOM metadata.
    
    Args:
        dcm: pydicom Dataset
        
    Returns:
        Dictionary with metadata
    """
    metadata = {
        'modality': getattr(dcm, 'Modality', 'UNKNOWN'),
        'window_center': getattr(dcm, 'WindowCenter', None),
        'window_width': getattr(dcm, 'WindowWidth', None),
        'patient_position': getattr(dcm, 'PatientPosition', None),
        'slice_thickness': getattr(dcm, 'SliceThickness', None),
        'pixel_spacing': getattr(dcm, 'PixelSpacing', None),
        'rows': getattr(dcm, 'Rows', None),
        'columns': getattr(dcm, 'Columns', None),
        'bits_stored': getattr(dcm, 'BitsStored', None),
        'rescale_intercept': getattr(dcm, 'RescaleIntercept', 0),
        'rescale_slope': getattr(dcm, 'RescaleSlope', 1),
    }
    
    # Handle WindowCenter/WindowWidth as lists (take first value)
    if isinstance(metadata['window_center'], (list, tuple)):
        metadata['window_center'] = float(metadata['window_center'][0])
    elif metadata['window_center'] is not None:
        metadata['window_center'] = float(metadata['window_center'])
    
    if isinstance(metadata['window_width'], (list, tuple)):
        metadata['window_width'] = float(metadata['window_width'][0])
    elif metadata['window_width'] is not None:
        metadata['window_width'] = float(metadata['window_width'])
    
    return metadata


def normalize_dicom_slice_mlx(
    slice_mx: mx.array,
    window_center: Optional[float] = None,
    window_width: Optional[float] = None,
    rescale_intercept: float = 0,
    rescale_slope: float = 1,
    modality: str = "CT"
) -> mx.array:
    """
    Apply DICOM windowing and normalization using MLX operations.
    
    Args:
        slice_mx: MLX array of shape (H, W)
        window_center: DICOM window center (level)
        window_width: DICOM window width
        rescale_intercept: DICOM rescale intercept
        rescale_slope: DICOM rescale slope
        modality: DICOM modality (CT, MRI, etc.)
        
    Returns:
        mx.array normalized to [0, 255] uint8 range
    """
    # Apply rescale slope and intercept (Hounsfield units for CT)
    slice_mx = slice_mx * rescale_slope + rescale_intercept
    
    # Apply windowing if provided
    if window_center is not None and window_width is not None:
        # Window/Level transformation
        lower = window_center - (window_width / 2)
        upper = window_center + (window_width / 2)
        
        # Clip to window range
        slice_mx = mx.clip(slice_mx, lower, upper)
        
        # Normalize to [0, 255]
        slice_mx = (slice_mx - lower) / (upper - lower) * 255.0
    else:
        # Auto-normalize to [0, 255] based on min/max
        min_val = mx.min(slice_mx)
        max_val = mx.max(slice_mx)
        
        if max_val > min_val:
            slice_mx = (slice_mx - min_val) / (max_val - min_val) * 255.0
        else:
            slice_mx = mx.zeros_like(slice_mx)
    
    # Convert to uint8
    slice_mx = mx.clip(slice_mx, 0, 255)
    slice_mx = slice_mx.astype(mx.uint8)
    
    # Evaluate to materialize
    mx.eval(slice_mx)
    
    return slice_mx


def mlx_slice_to_pil(
    slice_mx: mx.array,
    window_center: Optional[float] = None,
    window_width: Optional[float] = None,
    rescale_intercept: float = 0,
    rescale_slope: float = 1,
    modality: str = "CT"
) -> Image.Image:
    """
    Convert single MLX array slice to PIL Image.
    
    Args:
        slice_mx: MLX array of shape (H, W) or (H, W, C)
        window_center: Optional DICOM window center
        window_width: Optional DICOM window width
        rescale_intercept: DICOM rescale intercept
        rescale_slope: DICOM rescale slope
        modality: DICOM modality
        
    Returns:
        PIL Image in RGB mode
    """
    # Normalize if needed (for DICOM data)
    if slice_mx.dtype != mx.uint8:
        slice_mx = normalize_dicom_slice_mlx(
            slice_mx,
            window_center=window_center,
            window_width=window_width,
            rescale_intercept=rescale_intercept,
            rescale_slope=rescale_slope,
            modality=modality
        )
    
    # Convert to NumPy for PIL (minimal NumPy usage at I/O boundary)
    slice_np = np.array(slice_mx)
    
    # Create PIL Image
    if slice_np.ndim == 2:
        # Grayscale - convert to RGB
        image = Image.fromarray(slice_np, mode='L')
        image = image.convert('RGB')
    elif slice_np.ndim == 3:
        # Already has channels
        if slice_np.shape[2] == 1:
            image = Image.fromarray(slice_np[:, :, 0], mode='L')
            image = image.convert('RGB')
        elif slice_np.shape[2] == 3:
            image = Image.fromarray(slice_np, mode='RGB')
        else:
            # Take first 3 channels
            image = Image.fromarray(slice_np[:, :, :3], mode='RGB')
    else:
        raise ValueError(f"Unsupported slice shape: {slice_np.shape}")
    
    return image


def get_default_window_for_modality(modality: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Get default window/level settings for common modalities.
    
    Args:
        modality: DICOM modality string
        
    Returns:
        Tuple of (window_center, window_width) or (None, None)
    """
    defaults = {
        'CT': {
            'soft_tissue': (40, 400),
            'lung': (-600, 1500),
            'bone': (400, 1800),
            'brain': (40, 80),
        },
        'MRI': (None, None),  # MRI varies too much
        'CR': (None, None),   # Computed Radiography
        'DX': (None, None),   # Digital Radiography
        'US': (None, None),   # Ultrasound
    }
    
    # Return soft tissue window for CT, None for others
    if modality == 'CT':
        return defaults['CT']['soft_tissue']
    else:
        return (None, None)

import time
from functools import partial

from typing import Dict, List, Optional
import PIL
from PIL import Image
import numpy as np
import mlx.core as mx

from sam3.model import box_ops
from sam3.model.data_misc import FindStage, interpolate
from sam3.medical_utils import (
    preprocess_medical_image,
    MedicalModalityConfig,
    get_medical_prompt_suggestions
)


def transform(image_path_or_pil, resolution, modality=None):
    if isinstance(image_path_or_pil, str):
        img = Image.open(image_path_or_pil).convert("RGB")
    else:
        img = image_path_or_pil.convert("RGB")
    
    # Apply medical preprocessing if modality specified
    if modality and modality != "general":
        img_np, config = preprocess_medical_image(img, modality)
        img = Image.fromarray((img_np * 127.5 + 127.5).astype(np.uint8))
    
    img = img.resize((resolution, resolution), resample=Image.Resampling.LANCZOS)
    img_np = np.array(img).astype(np.float32) / 255.0 # [H, W, C]

    img_np = (img_np - 0.5) / 0.5

    return mx.array(img_np).transpose(2, 0, 1)  # [H, W, C] -> [C, H, W]

class Sam3Processor:
    def __init__(
        self,
        model,
        resolution=1008,
        confidence_threshold=0.5,
        modality: Optional[str] = None
    ):
        self.model = model
        self.resolution = resolution
        self.modality = modality or "general"
        
        # Use modality-specific threshold if available
        if modality:
            config = MedicalModalityConfig.get_config(modality)
            self.confidence_threshold = config.get("confidence_threshold", confidence_threshold)
            self.nms_threshold = config.get("nms_threshold", 0.5)
        else:
            self.confidence_threshold = confidence_threshold
            self.nms_threshold = 0.5
        
        self.transform = partial(transform, resolution=self.resolution, modality=self.modality)


        self.find_stage = FindStage(
            img_ids=mx.array([0], dtype=mx.int64),
            text_ids=mx.array([0], dtype=mx.int64),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

   
    def set_image(self, image, state=None):
        if state is None:
            state = {}
        
        if isinstance(image, PIL.Image.Image):
            width, height = image.size
        # elif isinstance(image, (mx.array, np.ndarray)):
        #     height, width = image.shape[-2:]
        else:
            raise ValueError("Image must be a PIL image")
        
        image = self.transform(image)[None]

        state["original_height"] = height
        state["original_width"] = width
        start = time.perf_counter()
        state["backbone_out"] = self.model.backbone.call_image(image)
        mx.eval(state)
        second = time.perf_counter()
        print(f"Backbone pass took {second - start:.2f} Seconds")
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )
        return state

    def set_image_batch(self, iamges: List[np.ndarray], state=None):
        pass

    def set_text_prompt(self, prompt: str, state: Dict):
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")
        
        text_outputs = self.model.backbone.call_text([prompt])
        # will erase the previous text prompt if any
        state["backbone_out"].update(text_outputs)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()
        return self._call_grounding(state)

    def add_geometric_prompt(self, box: List, label: bool, state: Dict):
        """Adds a box prompt and run the inference.
        The image needs to be set, but not necessarily the text prompt.
        The box is assumed to be in [center_x, center_y, width, height] format and normalized in [0, 1] range.
        The label is True for a positive box, False for a negative box.
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        if "language_features" not in state["backbone_out"]:
            # Looks like we don't have a text prompt yet. This is allowed, but we need to set the text prompt to "visual" for the model to rely only on the geometric prompt
            dummy_text_outputs = self.model.backbone.call_text(
                ["visual"]
            )
            state["backbone_out"].update(dummy_text_outputs)

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        # adding a batch and sequence dimension
        boxes = mx.array(box, dtype=mx.float32).reshape(1, 1, 4)
        labels = mx.array([label], dtype=mx.bool_).reshape(1, 1)
        state["geometric_prompt"].append_boxes(boxes, labels)

        return self._call_grounding(state)

    def add_point_prompt(self, point: List, label: bool, state: Dict):
        """Adds a point prompt and run the inference.
        The image needs to be set, but not necessarily the text prompt.
        The point is assumed to be in [x, y] format and normalized in [0, 1] range.
        The label is True for a positive point, False for a negative point.
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before add_point_prompt")

        if "language_features" not in state["backbone_out"]:
            # No text prompt yet - use "visual" to rely only on geometric prompt
            dummy_text_outputs = self.model.backbone.call_text(
                ["visual"]
            )
            state["backbone_out"].update(dummy_text_outputs)

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        # adding a batch and sequence dimension
        points = mx.array(point, dtype=mx.float32).reshape(1, 1, 2)
        labels = mx.array([label], dtype=mx.bool_).reshape(1, 1)
        state["geometric_prompt"].append_points(points, labels)

        return self._call_grounding(state)

    def reset_all_prompts(self, state: Dict):
        """Removes all the prompts and results"""
        if "backbone_out" in state:
            backbone_keys_to_del = [
                "language_features",
                "language_mask",
                "language_embeds",
            ]
            for key in backbone_keys_to_del:
                if key in state["backbone_out"]:
                    del state["backbone_out"][key]

        keys_to_del = ["geometric_prompt", "boxes", "masks", "masks_logits", "scores"]
        for key in keys_to_del:
            if key in state:
                del state[key]

    def set_confidence_threshold(self, threshold: float, state=None):
        """Update confidence threshold."""
        self.confidence_threshold = threshold
    
    def set_modality(self, modality: str):
        """
        Set medical imaging modality and update thresholds accordingly.
        
        Args:
            modality: Medical imaging modality (ct, mri, xray, ultrasound, etc.)
        """
        self.modality = modality
        config = MedicalModalityConfig.get_config(modality)
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.nms_threshold = config.get("nms_threshold", 0.5)
        self.transform = partial(transform, resolution=self.resolution, modality=self.modality)
        print(f"Set modality to {modality}: confidence={self.confidence_threshold}, nms={self.nms_threshold}")
    
    def get_medical_suggestions(self) -> List[str]:
        """Get suggested medical prompts for current modality."""
        return get_medical_prompt_suggestions(self.modality)

    def set_image_mlx(self, image_mlx: mx.array, state=None):
        """Set image from an MLX array (avoids PIL roundtrip for volume slices).
        
        Args:
            image_mlx: MLX array of shape (H, W) grayscale or (H, W, 3) RGB,
                       already in uint8 [0,255] or float [0,1] range.
            state: Optional existing state dict.
        """
        if state is None:
            state = {}
        
        image_tensor, height, width = self._prepare_slice_tensor(image_mlx)
        
        state["original_height"] = height
        state["original_width"] = width
        
        start = time.perf_counter()
        state["backbone_out"] = self.model.backbone.call_image(image_tensor)
        mx.eval(state)
        elapsed = time.perf_counter() - start
        print(f"Backbone pass (MLX) took {elapsed:.2f} Seconds")
        
        return state

    def _prepare_slice_tensor(self, slice_mlx: mx.array):
        """Prepare a single slice as a model-ready (1, 3, H, W) tensor.
        
        Returns (image_tensor, height, width).
        """
        if slice_mlx.ndim == 2:
            height, width = slice_mlx.shape
        else:
            height, width = slice_mlx.shape[:2]
        
        # Convert to float32 [0,1] if needed
        if slice_mlx.dtype == mx.uint8:
            img = slice_mlx.astype(mx.float32) / 255.0
        elif slice_mlx.dtype in (mx.int16, mx.int32, mx.uint16):
            min_val = mx.min(slice_mlx).item()
            max_val = mx.max(slice_mlx).item()
            if max_val > min_val:
                img = (slice_mlx.astype(mx.float32) - min_val) / (max_val - min_val)
            else:
                img = mx.zeros_like(slice_mlx, dtype=mx.float32)
        else:
            img = slice_mlx
        
        # Ensure 3 channels
        if img.ndim == 2:
            img = mx.stack([img, img, img], axis=-1)
        elif img.shape[-1] == 1:
            img = mx.repeat(img, repeats=3, axis=-1)
        
        # Resize to model resolution
        import mlx.nn as nn_mod
        img_batch = img[None]
        scale_h = self.resolution / height
        scale_w = self.resolution / width
        upsample = nn_mod.Upsample(scale_factor=(scale_h, scale_w), mode="linear", align_corners=False)
        img_resized = upsample(img_batch)[0]
        
        # Normalize to [-1, 1] and convert to (1, 3, H, W)
        img_resized = (img_resized - 0.5) / 0.5
        image_tensor = img_resized.transpose(2, 0, 1)[None]
        
        return image_tensor, height, width

    def propagate_to_volume(
        self,
        volume_mlx: mx.array,
        source_slice: int,
        source_state: Dict,
        direction: str = "both",
    ) -> Dict[int, Dict]:
        """Propagate segmentation from source slice to adjacent slices.
        
        Pre-computes backbone features for all target slices, then runs
        the lightweight grounding (encoder+decoder) per slice using
        centroid prompts from the previous slice.
        
        Args:
            volume_mlx: MLX array of shape (num_slices, H, W) or (num_slices, H, W, C)
            source_slice: Index of the already-segmented slice
            source_state: State dict from the source slice (must contain masks)
            direction: "forward", "backward", or "both"
            
        Returns:
            Dict mapping slice_index → state dict with masks
        """
        if "masks" not in source_state:
            raise ValueError("Source state must contain segmentation masks")
        
        num_slices = volume_mlx.shape[0]
        volume_states = {source_slice: source_state}
        
        # Determine which slices we need to process
        target_indices = []
        if direction in ("forward", "both"):
            target_indices.extend(range(source_slice + 1, num_slices))
        if direction in ("backward", "both"):
            target_indices.extend(range(source_slice - 1, -1, -1))
        
        if not target_indices:
            return volume_states
        
        # Phase 1: Pre-compute backbone features for all target slices
        print(f"  Pre-computing backbone features for {len(target_indices)} slices...")
        backbone_cache = {}
        start = time.perf_counter()
        for idx in target_indices:
            slice_mlx = volume_mlx[idx]
            image_tensor, height, width = self._prepare_slice_tensor(slice_mlx)
            backbone_out = self.model.backbone.call_image(image_tensor)
            mx.eval(backbone_out)
            backbone_cache[idx] = {
                "backbone_out": backbone_out,
                "original_height": height,
                "original_width": width,
            }
        elapsed = time.perf_counter() - start
        print(f"  Backbone pre-computation: {elapsed:.2f}s ({elapsed/len(target_indices):.2f}s/slice)")
        
        # Pre-compute text features once (shared across all slices)
        text_outputs = self.model.backbone.call_text(["visual"])
        mx.eval(text_outputs)
        
        def _get_mask_centroid(state):
            """Get normalized centroid of the largest mask."""
            masks = state["masks"]
            if len(masks.shape) < 2:
                return None
            mask = np.array(masks[0])
            if mask.ndim == 3:
                mask = mask[0]
            ys, xs = np.where(mask > 0.5)
            if len(ys) == 0:
                return None
            cy = float(np.mean(ys)) / mask.shape[0]
            cx = float(np.mean(xs)) / mask.shape[1]
            return [cx, cy]
        
        # Phase 2: Run lightweight grounding per slice using cached backbone features
        def _propagate_direction(start_slice, end_slice, step):
            prev_state = volume_states[start_slice]
            
            for idx in range(start_slice + step, end_slice, step):
                centroid = _get_mask_centroid(prev_state)
                if centroid is None:
                    break
                
                if idx not in backbone_cache:
                    break
                
                # Build state from cached backbone features
                cached = backbone_cache[idx]
                new_state = {
                    "original_height": cached["original_height"],
                    "original_width": cached["original_width"],
                    "backbone_out": dict(cached["backbone_out"]),
                }
                
                # Inject cached text features
                new_state["backbone_out"].update(text_outputs)
                
                # Set up geometric prompt with centroid point
                new_state["geometric_prompt"] = self.model._get_dummy_prompt()
                points = mx.array(centroid, dtype=mx.float32).reshape(1, 1, 2)
                labels = mx.array([True], dtype=mx.bool_).reshape(1, 1)
                new_state["geometric_prompt"].append_points(points, labels)
                
                # Run grounding only (encoder + decoder, no backbone)
                new_state = self._call_grounding(new_state)
                mx.eval(new_state["masks"])
                
                if "masks" not in new_state or len(new_state["scores"]) == 0:
                    break
                
                volume_states[idx] = new_state
                prev_state = new_state
                print(f"  Propagated to slice {idx}, found {len(new_state['scores'])} objects")
        
        start = time.perf_counter()
        if direction in ("forward", "both"):
            _propagate_direction(source_slice, num_slices, 1)
        if direction in ("backward", "both"):
            _propagate_direction(source_slice, -1, -1)
        elapsed = time.perf_counter() - start
        grounded = len(volume_states) - 1
        if grounded > 0:
            print(f"  Grounding pass: {elapsed:.2f}s ({elapsed/grounded:.2f}s/slice)")
        
        return volume_states

    def _call_grounding(self, state: Dict):
        outputs = self.model.call_grounding(
            backbone_out=state["backbone_out"],
            find_input=self.find_stage,
            geometric_prompt=state["geometric_prompt"],
            find_target=None
        )

        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = mx.sigmoid(out_logits)
        presence_score = mx.sigmoid(outputs["presence_logit_dec"])[:,None]
        out_probs = (out_probs * presence_score).squeeze(-1)

        keep = out_probs > self.confidence_threshold
        mask_np = np.array(keep[0])
        indices = mx.array(mask_np.nonzero()[0])
        out_probs = out_probs[0][indices]
        # out_probs = out_probs[keep]
        out_masks = out_masks[0][indices]
        out_bbox = out_bbox[0][indices]
        seg_mask = outputs['semantic_seg']

        # convert box to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        img_h = state["original_height"]
        img_w = state["original_width"]
        scale_fct = mx.array([img_w, img_h, img_w, img_h])
        boxes = boxes * scale_fct[None, :]

        interpolator = partial(interpolate,
            size=(img_h, img_w),
            mode="bilinear",
            align_corners=False,
        )
        out_masks = interpolator(out_masks[:, None])
        out_masks = mx.sigmoid(out_masks)

        seg_mask = interpolator(seg_mask)

        state["semantic_seg"] = seg_mask
        state["mask_logits"] = out_masks
        state["masks"] = out_masks > 0.5
        state["boxes"] = boxes
        state["scores"] = out_probs
        return state
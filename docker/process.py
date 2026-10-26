#!/usr/bin/env python3
"""
process.py
==========

Inference entry point for the competition Docker container
(18-class CT abnormality classification).

For every CT volume found in /input the script:
  1. Reads the scan (.mha / .nii / .nii.gz) with SimpleITK, clips intensities
     to [-1000, 1000] HU, resamples to 1.5 x 0.75 x 0.75 mm spacing and scales
     values to [-1, 1].
  2. Centre-crops / pads the volume to 400 x 400 x 240 (pad value -1).
  3. Runs the ResNet-18 + ViViT fusion model (EnhancedFusionModel). The model
     keeps the middle 160 slices internally.
  4. Applies a sigmoid, then the per-class scale/shift calibration produced by
     scripts_vlm_challenge/model_calibration.py.
  5. Writes every probability to /output/results.json.

Container layout
----------------
```
/input                                       ← CT volumes (*.mha / *.nii / *.nii.gz)
/output                                      ← receives results.json

/opt/app/models/best_model.pth               ← EnhancedFusionModel weights
/opt/app/models/calibration_params.joblib    ← per-class (scale, shift) calibration
```

Note: `transformers`, `transformer_maskgit` and `ct_clip` are still imported at
start-up (left over from the CT-CLIP baseline), so they must be installed in the
image even though the fusion model does not use them.

Output schema
-------------
```json
{
  "name": "Generated probabilities",
  "type": "Abnormality Classification",
  "version": {"major": 1, "minor": 0},
  "predictions": [
    {
      "input_image_name": "<volume name>",
      "probabilities": {"Medical material": 0.00, "Arterial wall calcification": 0.42, …}
    }
  ]
}
```

The script runs on CUDA when available and falls back to CPU.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
from tqdm import tqdm

from PIL import Image

SLICE_DIR = Path("/output/mid_slices")
SLICE_DIR.mkdir(parents=True, exist_ok=True)

CLASSIFIER_PATH  = Path("/opt/app/models/best_model.pth")
CALIBRATION_PATH = Path("/opt/app/models/calibration_params.joblib")

import torch
import torch.nn as nn
from vit_pytorch.vivit import ViT
from torchvision import models
import joblib

# ---------------------------------------------------------------------------
#  HARD‑WIRED PATHS & CONSTANTS
# ---------------------------------------------------------------------------
VOL_DIR          = Path("/input")
OUT_FILE         = Path("/output/results.json")

TEXT_MODEL_PATH  = Path("/opt/app/models/BiomedVLP-CXR-BERT-specialized")

DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

HU_MIN, HU_MAX   = -1000, 1000
TARGET_SHAPE     = (480, 480, 240)            # (H, W, D)
TARGET_SPACING   = (1.5, 0.75, 0.75)           # (z, x, y) mm

PATHOLOGIES = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
    "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule", "Lung opacity",
    "Pulmonary fibrotic sequela", "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation", "Bronchiectasis",
    "Interlobular septal thickening",
]
NUM_CLASSES = len(PATHOLOGIES)

# ---------------------------------------------------------------------------
#  THIRD‑PARTY IMPORTS (fail fast if missing)
# ---------------------------------------------------------------------------
try:
    from transformers import BertTokenizer, BertModel
    from transformer_maskgit import CTViT
    from ct_clip import CTCLIP
except Exception as e:  # pragma: no cover
    raise SystemExit(f"✗ Required library missing: {e}")

# ---------------------------------------------------------------------------
#  MODEL DEFINITIONS
# ---------------------------------------------------------------------------

def sigmoid(tensor):
    return 1 / (1 + torch.exp(-tensor))


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))
        self.dim = dim

    def forward(self, x):
        # Make sure we're normalizing over the correct dimension
        return F.layer_norm(x, (self.dim,), self.gamma, self.beta)


class SimpleVideoClassifier(nn.Module):
    def __init__(self, num_classes=18, num_frames=160):
        super().__init__()

        # initialize ViViT with parameters similar to SmallViT
        self.vivit = ViT(
            image_size=400,
            frames=num_frames,  # number of frames (slices)
            image_patch_size=16,
            frame_patch_size=10,  # Process X frames at once
            num_classes=num_classes,
            dim=384,
            spatial_depth=6,  # spatial transformer depth
            temporal_depth=1,  # temporal transformer depth
            heads=8,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1,
            variant='factorized_encoder',
        )

    def forward(self, x):
        # Input: [batch_size, channels, frames, height, width]
        B, C, T, H, W = x.shape

        # Take middle 160 slices if more are provided

        if T > 160:
           mid_point = T // 2
           start_idx = mid_point - 80
           end_idx = mid_point + 80
           x = x[:, :, start_idx:end_idx, :, :]

        # Handle single-channel input
        if C == 1:
            x = x.repeat(1, 3, 1, 1, 1)

        return self.vivit(x)

class EnhancedFusionModel(nn.Module):
    def __init__(self, base_2d_model, base_video_model, num_classes=18):
        super(EnhancedFusionModel, self).__init__()

        # Store the base models
        self.base_2d_model = base_2d_model
        self.base_video_model = base_video_model

        # Get feature dimensions from 2D model
        self.features_2d = self.base_2d_model.fc.in_features

        # Replace the final classification layers
        self.base_2d_model.fc = nn.Identity()

        # For the SimpleVideoClassifier which contains a ViT model
        # Access the dimension used in ViViT
        self.features_video = base_video_model.vivit.mlp_head.in_features

        # Store the original classifier and replace it with Identity
        self.original_video_classifier = base_video_model.vivit.mlp_head
        base_video_model.vivit.mlp_head = nn.Identity()

        # Feature fusion components
        self.projection_video = nn.Linear(self.features_video, self.features_2d)

        # Learnable fusion parameter
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Attention fusion
        self.fusion_attention = nn.Sequential(
            nn.Linear(self.features_2d * 2, self.features_2d),
            nn.ReLU(),
            nn.Linear(self.features_2d, 2),
            nn.Softmax(dim=1)
        )

        # Final classifier
        self.classifier = nn.Linear(self.features_2d, num_classes)

    def forward(self, x):
        # x is a 5D tensor: [batch_size, channels, depth, height, width]
        batch_size, channels, depth, height, width = x.shape

        # Take middle 160 slices if more are provided
        if depth > 160:
           mid_point = depth // 2
           start_idx = mid_point - 80
           end_idx = mid_point + 80
           x = x[:, :, start_idx:end_idx, :, :]
           depth = 160

        # Process with video model
        if channels == 1:
            x_video = x.repeat(1, 3, 1, 1, 1)  # ViViT expects 3 channels
        else:
            x_video = x

        # Extract features from the video model (ViViT returns embedding directly when mlp_head is Identity)
        video_features = self.base_video_model.vivit(x_video)  # [batch_size, features_video]

        # Project video features to match 2D feature dimension
        video_features = self.projection_video(video_features)

        # Process with 2D model (slice by slice)
        x_2d = x.permute(0, 2, 1, 3, 4)  # [batch_size, depth, channels, height, width]
        x_2d = x_2d.reshape(batch_size * depth, channels, height, width)

        # Process all slices at once
        features_2d = self.base_2d_model(x_2d)  # [batch_size*depth, features_2d]

        # Reshape back and aggregate across slices
        features_2d = features_2d.view(batch_size, depth, -1)  # [batch_size, depth, features_2d]
        features_2d = torch.mean(features_2d, dim=1)  # [batch_size, features_2d]

        # Concatenate features for attention mechanism
        combined = torch.cat([features_2d, video_features], dim=1)
        attention_weights = self.fusion_attention(combined)

        # Apply attention weights
        fused_features = (features_2d * attention_weights[:, 0].unsqueeze(1) +
                          video_features * attention_weights[:, 1].unsqueeze(1))

        # Final classification
        output = self.classifier(fused_features)

        return output

def initialize_fusion_model(num_classes=18, pretrained_2d=False):
    """
    Initialize the enhanced fusion model combining 2D ResNet and ViViT video classifier.
    """
    # Initialize 2D ResNet
    base_2d_model = models.resnet18(weights='IMAGENET1K_V1' if pretrained_2d else None)
    # Modify first layer to accept single-channel input
    base_2d_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Initialize ViViT-based video classifier
    base_video_model = SimpleVideoClassifier(num_classes=num_classes, num_frames=160)

    # Create fusion model
    fusion_model = EnhancedFusionModel(base_2d_model, base_video_model, num_classes)

    return fusion_model


def build_fusion_model() -> EnhancedFusionModel:
    """Constructs fusion model with pretrained weights."""
    # Initialize 2D ResNet
    base_2d_model = models.resnet18()
    # Modify first layer to accept single-channel input
    base_2d_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Initialize ViViT-based video classifier
    base_video_model = SimpleVideoClassifier(num_classes=NUM_CLASSES, num_frames=160)

    # Create fusion model
    model = EnhancedFusionModel(base_2d_model, base_video_model, NUM_CLASSES)

    # Load state dictionary
    model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model


# ---------------------------------------------------------------------------
#  PRE‑PROCESSING
# ---------------------------------------------------------------------------

def resize_vol(vol: torch.Tensor, spacing: tuple[float, float, float]) -> torch.Tensor:
    """Trilinear resampling to TARGET_SPACING."""
    z, x, y = spacing
    scaling = [spacing[i] / TARGET_SPACING[i] for i in range(3)]
    D, H, W = vol.shape[2:]
    new_shape = [int(D * scaling[0]), int(H * scaling[1]), int(W * scaling[2])]
    return F.interpolate(vol, size=new_shape, mode="trilinear", align_corners=False)


def preprocess_volume(path: Path) -> torch.Tensor:
    """HU clipping, resampling, normalisation to [-1, 1] and centre crop/pad to 400 x 400 x 240."""
    itk  = sitk.ReadImage(str(path))
    img  = sitk.GetArrayFromImage(itk).astype("float32")  # z, y, x
    print(img.max())
    print(img.min())
    vol  = np.clip(img, HU_MIN, HU_MAX)
    print(vol.shape)
    t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    sx, sy, sz = itk.GetSpacing()                        # spacing (x,y,z)
    t = resize_vol(t, (sz, sx, sy))                      # need (z,x,y)
    print(t.shape)
    tensor = t[0, 0].permute(1, 2, 0)                    # (H,W,D)
    tensor = tensor / 1000.0                             # HU → [−1,1]

    slices = []

    target_shape = (400, 400, 240)

    # Extract dimensions
    h, w, d = tensor.shape

    # Calculate cropping/padding values for height, width, and depth
    dh, dw, dd = target_shape
    h_start = max((h - dh) // 2, 0)
    h_end = min(h_start + dh, h)
    w_start = max((w - dw) // 2, 0)
    w_end = min(w_start + dw, w)
    d_start = max((d - dd) // 2, 0)
    d_end = min(d_start + dd, d)

    # Crop or pad the tensor
    tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

    pad_h_before = (dh - tensor.size(0)) // 2
    pad_h_after = dh - tensor.size(0) - pad_h_before

    pad_w_before = (dw - tensor.size(1)) // 2
    pad_w_after = dw - tensor.size(1) - pad_w_before

    pad_d_before = (dd - tensor.size(2)) // 2
    pad_d_after = dd - tensor.size(2) - pad_d_before

    tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)

    tensor = tensor.permute(2, 1, 0)

    tensor = tensor.unsqueeze(0).unsqueeze(0)

    print(tensor.shape)
    print(tensor.max())
    print(tensor.min())
    return tensor


# ---------------------------------------------------------------------------
#  INFERENCE
# ---------------------------------------------------------------------------


def apply_calibration(probabilities, calibration_file):
    """
    Apply calibration to new probability predictions.

    Args:
        probabilities: Numpy array of shape (n_samples, n_classes)
        calibration_file: Path to saved calibration parameters

    Returns:
        Calibrated probabilities
    """
    calibration_params = joblib.load(calibration_file)
    calibrated_probs = probabilities.copy()

    for class_idx, (scale, shift) in calibration_params.items():
        calibrated_probs[:, class_idx] = np.minimum(1.0, probabilities[:, class_idx] * scale + shift)

    return calibrated_probs

def predict(model: EnhancedFusionModel, vol_tensor: torch.Tensor) -> Dict[str, float]:
    model.eval()
    vol_tensor = vol_tensor.to(DEVICE)
    with torch.no_grad():
        logits = model(vol_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    # calibration
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)

    calibrated_probs = apply_calibration(probs, CALIBRATION_PATH)
    probs = calibrated_probs[0]

    results = {p: float(prob) for p, prob in zip(PATHOLOGIES, probs)}
    return results

# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

def main():
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

    vols = sorted([p for p in VOL_DIR.iterdir() if p.suffix.lower() in {".mha", ".nii", ".nii.gz"}])
    if not vols:
        raise SystemExit(f"✗ No CT volumes found in {VOL_DIR}")

    classifier = build_fusion_model()

    predictions: List[Dict[str, str | Dict[str, float]]] = []
    for v in tqdm(vols, desc="Volumes", unit="scan"):
        vol_tensor = preprocess_volume(v)
        scores     = predict(classifier, vol_tensor)
        predictions.append({"input_image_name": v.name.replace(".mha","").replace(".nii",""), "probabilities": scores})
        tqdm.write("✓ " + v.name)

    result = {
        "name": "Generated probabilities",
        "type": "Abnormality Classification",
        "version": {"major": 1, "minor": 0},
        "predictions": predictions,
    }
    print(result)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w") as f:
        json.dump(result, f, indent=2)
    print("Saved →", OUT_FILE)


if __name__ == "__main__":
    main()

"""
model_calibration.py
--------------------
Post-hoc probability calibration for the submitted fusion model. Produces the
calibration_params.joblib file used by the Docker process.py.

Steps
  1. Loads the trained EnhancedFusionModel (ResNet-18 + ViViT, dim=384,
     160 slices) from CLASSIFIER_PATH.
  2. Runs it on the CT-RATE validation set and collects sigmoid probabilities.
  3. For each class, grid-searches a scale and shift so that
         p_calibrated = min(1, p * scale + shift)
     maximises F1 at a fixed 0.5 threshold.
  4. Saves calibration_params.joblib ({class_idx: (scale, shift)}), before/after
     metric CSVs, confusion matrices, probability-distribution plots,
     calibration curves and a small apply_calibration.py helper to output_dir.

Usage: edit CLASSIFIER_PATH and the data/output paths, then
    python model_calibration.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
import joblib
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
import seaborn as sns
from torch.utils.data import DataLoader
import sys


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.isotonic import IsotonicRegression
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import joblib


from model_architecture_helper_functions import *
import math
from einops import rearrange
from einops.layers.torch import Rearrange

import torch
import torch.nn as nn
from vit_pytorch.vivit import ViT
from torchvision import models

import seaborn as sns
import os


# Constants
CLASSIFIER_PATH = Path("/labs/khatrilab/maestri/classes/cs231n/figures/vlm_challenge/5_epoch_400px_160_slices_video_ResNet_NEW_384dim/best_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 18


CUSTOM_THRESHOLDS = [0.25, 0.32, 0.27, 0.19, 0.35, 0.21, 0.32, 0.28, 0.3, 
                    0.33, 0.31,0.26, 0.38, 0.22, 0.15, 0.27, 0.13, 0.18]

# Pathology class names (same order as the model outputs)
PATHOLOGY_NAMES = ["Medical material", "Arterial wall calcification", "Cardiomegaly", 
                  "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
                  "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule", "Lung opacity",
                  "Pulmonary fibrotic sequela", "Pleural effusion", "Mosaic attenuation pattern",
                  "Peribronchial thickening", "Consolidation", "Bronchiectasis", "Interlobular septal thickening"]


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

def find_optimal_calibration_params(val_probs, val_labels, class_idx):
    """
    Find optimal scaling and shifting parameters for a given class
    that maximizes F1 score at 0.5 threshold.
    """
    best_f1 = 0
    best_scale = 1.0
    best_shift = 0.0

    # Grid search for optimal parameters
    for scale in np.arange(0.5, 3.0, 0.1):
        for shift in np.arange(0.0, 0.5, 0.02):
            # Apply scaling and shifting
            calibrated_probs = np.minimum(1.0, val_probs[:, class_idx] * scale + shift)

            # Check performance at 0.5 threshold
            preds = (calibrated_probs >= 0.5).astype(int)
            f1 = f1_score(val_labels[:, class_idx], preds)

            if f1 > best_f1:
                best_f1 = f1
                best_scale = scale
                best_shift = shift

    return best_scale, best_shift, best_f1

def calibrate_probabilities(probs, calibration_params):
    """
    Apply scaling and shifting to calibrate probabilities.
    """
    calibrated_probs = probs.copy()

    for class_idx, (scale, shift) in calibration_params.items():
        calibrated_probs[:, class_idx] = np.minimum(1.0, probs[:, class_idx] * scale + shift)

    return calibrated_probs


def load_validation_data(config, data_dirs, metadata_file, val_csv, batch_size=8):
    """
    Load validation data for calibration fitting
    
    Args:
        config: Configuration dictionary with model parameters
        data_dirs: Base directory for data
        metadata_file: Path to metadata file
        val_csv: Path to validation labels CSV
        batch_size: Batch size for data loading
        
    Returns:
        DataLoader for validation data
    """
    # Set parameters from config
    resize_dim = config.get('resize_dim', 224)
    num_slices = config.get('num_slices', 100)
    slice_sampling_method = config.get('slice_sampling', 'center')

    # Create validation dataset
    val_dataset = CTReportDataset(
        data_folder=data_dirs + "valid/",
        csv_file=metadata_file + "validation_reports.csv",
        label_file=val_csv,
        resize_dim=resize_dim,
        num_slices=num_slices,
        slice_sampling=slice_sampling_method
    )

    print(f"Validation dataset created with {len(val_dataset)} samples")

    # Create DataLoader without distributed sampler since this is for Platt scaling
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep order consistent for calibration
        num_workers=4,
        pin_memory=True
    )

    return val_loader

def plot_calibration_curves(true_labels, original_probs, calibrated_probs, label_columns, save_path):
    """
    Plot calibration curves for original and calibrated probabilities.
    """
    from sklearn.calibration import calibration_curve

    num_classes = true_labels.shape[1]  # 18 classes

    # Define a grid that can accommodate 18 classes
    grid_cols = 5  # Using 5 columns allows for 5x4=20 plots
    grid_rows = 4

    plt.figure(figsize=(20, 16))  # Larger figure to accommodate more plots

    for i in range(num_classes):
        if i < grid_cols * grid_rows:  # Ensure we don't exceed the grid
            plt.subplot(grid_rows, grid_cols, i+1)

            # Original calibration curve
            prob_true, prob_pred = calibration_curve(true_labels[:, i], original_probs[:, i], n_bins=10)
            plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Original')

            # Calibrated curve
            prob_true, prob_pred = calibration_curve(true_labels[:, i], calibrated_probs[:, i], n_bins=10)
            plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Calibrated')

            # Perfect calibration reference line
            plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')

            # Use class labels if provided
            class_name = label_columns[i] if label_columns else f'Class {i}'
            plt.title(class_name, fontsize=10)

            if i % grid_cols == 0:
                plt.ylabel('Fraction of positives')
            if i >= num_classes - grid_cols:
                plt.xlabel('Mean predicted probability')
            plt.legend(loc='best', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrices(all_labels, all_preds, thresholds, label_columns, split, output_dir):
    """
    Plot confusion matrices for each class based on your existing function.
    """
    n_classes = all_labels.shape[1]

    # Apply thresholds to get binary predictions
    binary_preds = np.zeros_like(all_preds)
    for i in range(n_classes):
        binary_preds[:, i] = (all_preds[:, i] >= thresholds[i]).astype(int)

    # Calculate rows and columns for the subplot grid
    n_cols = 5
    n_rows = (n_classes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.ravel()

    for i in range(n_classes):
        true_binary = all_labels[:, i]
        pred_binary = binary_preds[:, i]

        cm = confusion_matrix(true_binary, pred_binary)

        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
        axes[i].set_title(f'{label_columns[i]}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')

    for i in range(n_classes, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{split}_confusion_matrices.pdf'))
    plt.close()

def plot_probability_distributions(val_labels, val_probs, calibrated_probs, label_columns, output_dir):
    """
    Plot histograms of probability distributions before and after calibration.
    """
    os.makedirs(os.path.join(output_dir, 'probability_distributions'), exist_ok=True)

    for i, class_name in enumerate(label_columns):
        plt.figure(figsize=(15, 6))

        # Get positive and negative examples
        pos_mask = val_labels[:, i] == 1
        neg_mask = val_labels[:, i] == 0

        # Plot original probabilities
        plt.subplot(1, 2, 1)
        plt.hist(val_probs[pos_mask, i], bins=20, alpha=0.7, label='Positive samples', color='green')
        plt.hist(val_probs[neg_mask, i], bins=20, alpha=0.7, label='Negative samples', color='red')
        plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
        plt.title(f'{class_name} - Original Probabilities')
        plt.xlabel('Probability')
        plt.ylabel('Count')
        plt.legend()

        # Plot calibrated probabilities
        plt.subplot(1, 2, 2)
        plt.hist(calibrated_probs[pos_mask, i], bins=20, alpha=0.7, label='Positive samples', color='green')
        plt.hist(calibrated_probs[neg_mask, i], bins=20, alpha=0.7, label='Negative samples', color='red')
        plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
        plt.title(f'{class_name} - Calibrated Probabilities')
        plt.xlabel('Probability')
        plt.ylabel('Count')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'probability_distributions', f'{class_name}_distribution.pdf'))
        plt.close()

def calculate_per_class_metrics(true_labels, preds, thresholds, label_columns):
    """
    Calculate detailed metrics for each class.
    """
    results = []

    for i, class_name in enumerate(label_columns):
        # Apply threshold for this class
        preds_binary = (preds[:, i] >= thresholds[i]).astype(int)

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(true_labels[:, i], preds_binary, labels=[0, 1]).ravel()

        # Calculate metrics
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Calculate AUROC
        try:
            auroc = roc_auc_score(true_labels[:, i], preds[:, i])
        except:
            auroc = 0.5  # Default for random classifier

        results.append({
            'class': class_name,
            'threshold': thresholds[i],
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'auroc': auroc
        })

    return pd.DataFrame(results)

def main():
    # Load configuration and paths (same as in your existing code)
    config = {
        'resize_dim': 400,  # Match your model's expected input size
        'num_slices': 160,
        'slice_sampling': 'center',
    }

    # Data directories (adjust to match your paths)
    dicom_dir = "/labs/khatrilab/maestri/classes/cs231n/data/ctrate/data_volumes/dataset/"
    metadata_file = "/labs/khatrilab/maestri/classes/cs231n/data/ctrate/"
    model_name = "vlm_challenge"
    output_dir = "/labs/khatrilab/maestri/classes/cs231n/figures/"+model_name+"/calibration/"
    val_csv = "/labs/khatrilab/maestri/classes/cs231n/data/ctrate/valid_predicted_labels.csv"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Label columns - adjust to match your actual class names
    label_columns = PATHOLOGY_NAMES

    # Classes that don't need calibration (already performing well)

    skip_classes = []

    # 1. Load your model
    print("Loading model...")
    model = build_fusion_model()

    # 2. Load validation data for calibration
    print("Loading validation data...")
    val_loader = load_validation_data(
        config=config,
        data_dirs=dicom_dir,
        metadata_file=metadata_file,
        val_csv=val_csv,
        batch_size=10
    )

    # 3. Collect validation predictions and labels
    print("Running model on validation data...")
    val_logits = []
    val_labels = []

    model.eval()
    with torch.no_grad():
        for videos, reports, labels in val_loader:
            videos = videos.to(DEVICE).float()
            labels = labels.float()  # Keep on CPU for now

            outputs = model(videos)
            val_logits.append(outputs.cpu().numpy())
            val_labels.append(labels.numpy())

    val_logits = np.vstack(val_logits)
    val_labels = np.vstack(val_labels)

    # 4. Convert logits to probabilities
    val_probs = 1 / (1 + np.exp(-val_logits))

    # 5. Evaluate original model with custom thresholds
    print("Evaluating original model with custom thresholds...")
    val_preds_custom = np.zeros_like(val_probs)
    for i in range(NUM_CLASSES):
        val_preds_custom[:, i] = (val_probs[:, i] >= CUSTOM_THRESHOLDS[i]).astype(int)

    # Calculate metrics for custom thresholds
    custom_f1 = f1_score(val_labels, val_preds_custom, average='macro')
    custom_auroc = roc_auc_score(val_labels, val_probs, average='macro')
    print(f"Original model with custom thresholds - F1: {custom_f1:.4f}, AUROC: {custom_auroc:.4f}")

    # 6. Evaluate original model with 0.5 thresholds
    print("Evaluating original model with 0.5 thresholds...")
    val_preds_default = np.zeros_like(val_probs)
    for i in range(NUM_CLASSES):
        val_preds_default[:, i] = (val_probs[:, i] >= 0.5).astype(int)

    # Calculate metrics for default thresholds
    default_f1 = f1_score(val_labels, val_preds_default, average='macro')
    print(f"Original model with 0.5 thresholds - F1: {default_f1:.4f}, AUROC: {custom_auroc:.4f}")

    # 7. Find optimal calibration parameters for each class
    print("Finding optimal calibration parameters...")
    calibration_params = {}
    calibration_results = []

    for i, class_name in enumerate(label_columns):
        # Skip classes that are already performing well
        if class_name in skip_classes:
            print(f"Skipping {class_name} - already performing well")
            calibration_params[i] = (1.0, 0.0)  # No adjustment
            continue

        # Find optimal parameters
        scale, shift, improved_f1 = find_optimal_calibration_params(val_probs, val_labels, i)
        calibration_params[i] = (scale, shift)

        # Calculate original F1 for this class
        original_preds = (val_probs[:, i] >= 0.5).astype(int)
        original_f1 = f1_score(val_labels[:, i], original_preds)

        print(f"Class {class_name}: scale={scale:.2f}, shift={shift:.2f}, F1: {original_f1:.4f} → {improved_f1:.4f}")

        calibration_results.append({
            'class': class_name,
            'scale': scale,
            'shift': shift,
            'original_f1': original_f1,
            'improved_f1': improved_f1,
            'improvement': improved_f1 - original_f1
        })

    # Save calibration parameters
    pd.DataFrame(calibration_results).to_csv(
        os.path.join(output_dir, 'calibration_parameters.csv'), 
        index=False
    )
    joblib.dump(calibration_params, os.path.join(output_dir, 'calibration_params.joblib'))

    # 8. Apply calibration
    print("Applying calibration...")
    calibrated_probs = calibrate_probabilities(val_probs, calibration_params)

    # 9. Evaluate with calibration and 0.5 thresholds
    print("Evaluating with calibration and 0.5 thresholds...")
    cal_preds = np.zeros_like(calibrated_probs)
    for i in range(NUM_CLASSES):
        cal_preds[:, i] = (calibrated_probs[:, i] >= 0.5).astype(int)

    # Calculate metrics after calibration
    cal_f1 = f1_score(val_labels, cal_preds, average='macro')
    cal_auroc = roc_auc_score(val_labels, calibrated_probs, average='macro')
    print(f"Calibrated model with 0.5 thresholds - F1: {cal_f1:.4f}, AUROC: {cal_auroc:.4f}")
    print(f"Improvement in F1: {cal_f1 - default_f1:.4f}")

    # 10. Calculate per-class metrics for original and calibrated probabilities
    print("Calculating per-class metrics...")
    original_metrics = calculate_per_class_metrics(
        val_labels, val_probs,
        [0.5] * len(label_columns),
        label_columns
    )

    calibrated_metrics = calculate_per_class_metrics(
        val_labels, calibrated_probs,
        [0.5] * len(label_columns),
        label_columns
    )

    # Save metrics to CSV
    original_metrics.to_csv(os.path.join(output_dir, 'original_metrics.csv'), index=False)
    calibrated_metrics.to_csv(os.path.join(output_dir, 'calibrated_metrics.csv'), index=False)

    # 11. Plot confusion matrices
    print("Plotting confusion matrices...")
    # Original model confusion matrices
    plot_confusion_matrices(
        val_labels, val_probs,
        [0.5] * len(label_columns),
        label_columns,
        'original',
        output_dir
    )

    # Calibrated model confusion matrices
    plot_confusion_matrices(
        val_labels, calibrated_probs,
        [0.5] * len(label_columns),
        label_columns,
        'calibrated',
        output_dir
    )

    # 12. Plot probability distributions before and after calibration
    print("Plotting probability distributions...")
    plot_probability_distributions(
        val_labels, val_probs, calibrated_probs,
        label_columns, output_dir
    )

    # 13. Plot calibration curves
    print("Plotting calibration curves...")
    plot_calibration_curves(
        val_labels, val_probs, calibrated_probs,
        label_columns,
        os.path.join(output_dir, 'calibration_curves.pdf')
    )

    # 14. Create comparison table showing improvements
    comparison = []
    for i, class_name in enumerate(label_columns):
        orig_metrics = original_metrics[original_metrics['class'] == class_name].iloc[0]
        calib_metrics = calibrated_metrics[calibrated_metrics['class'] == class_name].iloc[0]

        comparison.append({
            'class': class_name,
            'original_f1': orig_metrics['f1'],
            'calibrated_f1': calib_metrics['f1'],
            'f1_improvement': calib_metrics['f1'] - orig_metrics['f1'],
            'original_recall': orig_metrics['recall'],
            'calibrated_recall': calib_metrics['recall'],
            'recall_improvement': calib_metrics['recall'] - orig_metrics['recall'],
            'original_precision': orig_metrics['precision'],
            'calibrated_precision': calib_metrics['precision'],
            'precision_improvement': calib_metrics['precision'] - orig_metrics['precision'],
            'scale': calibration_params.get(i, (1.0, 0.0))[0],
            'shift': calibration_params.get(i, (1.0, 0.0))[1]
        })

    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(os.path.join(output_dir, 'calibration_comparison.csv'), index=False)

    # 15. Print summary statistics
    print("\nSummary of Calibration Results:")
    print(f"Original F1 (macro): {default_f1:.4f}")
    print(f"Calibrated F1 (macro): {cal_f1:.4f}")
    print(f"F1 Improvement: {cal_f1 - default_f1:.4f}")

    # Calculate average improvement in F1 across all classes
    avg_improvement = comparison_df['f1_improvement'].mean()
    print(f"Average class-wise F1 improvement: {avg_improvement:.4f}")

    # Show classes with largest improvements
    top_improvements = comparison_df.sort_values('f1_improvement', ascending=False)
    print("\nTop 3 most improved classes:")
    for _, row in top_improvements.head(3).iterrows():
        print(f"{row['class']}: {row['original_f1']:.4f} → {row['calibrated_f1']:.4f} (Δ: {row['f1_improvement']:.4f})")

    # 16. Save calibration model for future use
    print("\nSaving calibration parameters...")
    joblib.dump(calibration_params, os.path.join(output_dir, 'calibration_params.joblib'))

    # 17. Create a simple function to apply calibration to new data
    with open(os.path.join(output_dir, 'apply_calibration.py'), 'w') as f:
        f.write("""
import numpy as np
import joblib

def apply_calibration(probabilities, calibration_file):
    \"\"\"
    Apply calibration to new probability predictions.
    
    Args:
        probabilities: Numpy array of shape (n_samples, n_classes)
        calibration_file: Path to saved calibration parameters
        
    Returns:
        Calibrated probabilities
    \"\"\"
    calibration_params = joblib.load(calibration_file)
    calibrated_probs = probabilities.copy()
    
    for class_idx, (scale, shift) in calibration_params.items():
        calibrated_probs[:, class_idx] = np.minimum(1.0, probabilities[:, class_idx] * scale + shift)
    
    return calibrated_probs

# Example usage:
# raw_predictions = model.predict_proba(new_data)
# calibrated_predictions = apply_calibration(raw_predictions, 'calibration_params.joblib')
""")
    
    print(f"\nAll results saved to {output_dir}")
    print("Calibration complete!")

if __name__ == "__main__":
    main()

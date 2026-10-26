"""
platt_scaling.py
----------------
Earlier calibration experiment using Platt scaling (one logistic regression
per class on the raw logits). Run on the 100-slice version of the fusion model.
The submission uses the scale/shift method in model_calibration.py instead.

Steps
  1. Loads the trained fusion model and collects validation logits/labels.
  2. Splits the validation set 80/20, fits a PlattScaler on the 80% split and
     evaluates on the remaining 20%.
  3. Reports metrics with the original thresholds, 0.5 thresholds and
     F1-optimal thresholds after calibration.
  4. Saves the fitted scaler, optimal thresholds, metric CSVs, confusion
     matrices and calibration curves (example outputs are in platt_v1/).

Usage: edit CLASSIFIER_PATH and the data paths, then
    python platt_scaling.py
"""

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
CLASSIFIER_PATH = Path("/labs/khatrilab/maestri/classes/cs231n/figures/vlm_challenge/5_epoch_400px_100_slices_video_ResNet/best_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 18

# Per-class decision thresholds from the uncalibrated model
CUSTOM_THRESHOLDS = [0.21, 0.39, 0.32, 0.15, 0.29, 0.18, 0.27, 0.26, 0.29, 
                     0.29, 0.33, 0.23, 0.35, 0.19, 0.14, 0.27, 0.12, 0.17]

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
    def __init__(self, num_classes=18, num_frames=100):
        super().__init__()

        # initialize ViViT with parameters similar to SmallViT
        self.vivit = ViT(
            image_size=400,
            frames=num_frames,  # number of frames (slices)
            image_patch_size=16,
            frame_patch_size=10,  # Process X frames at once
            num_classes=num_classes,
            dim=256,
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

        # Take middle 100 slices if more are provided
        if T > 100:
            mid_point = T // 2
            start_idx = mid_point - 50
            end_idx = mid_point + 50
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

        # Take middle 100 slices if more are provided
        if depth > 100:
             mid_point = depth // 2
             start_idx = mid_point - 50
             end_idx = mid_point + 50
             x = x[:, :, start_idx:end_idx, :, :]
             depth = 100

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
    base_video_model = SimpleVideoClassifier(num_classes=num_classes, num_frames=100)

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
    base_video_model = SimpleVideoClassifier(num_classes=NUM_CLASSES, num_frames=100)

    # Create fusion model
    model = EnhancedFusionModel(base_2d_model, base_video_model, NUM_CLASSES)

    # Load state dictionary
    model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model


# Platt scaling: one logistic regression per class on the raw logits
class PlattScaler:
    def __init__(self, num_classes=18):
        self.num_classes = num_classes
        self.coefficients = []  # Will store (a, b) pairs for each class

    def fit(self, logits, targets):
        """
        Fit Platt scaling parameters manually
        
        Args:
            logits: Model predictions (before sigmoid) - shape (n_samples, n_classes)
            targets: Ground truth binary labels - shape (n_samples, n_classes)
        """
        from sklearn.linear_model import LogisticRegression

        self.coefficients = []

        for i in range(self.num_classes):
            # Fit logistic regression: P(y=1) = sigmoid(a*logit + b)
            lr = LogisticRegression(fit_intercept=True, max_iter=1000)
            lr.fit(logits[:, i].reshape(-1, 1), targets[:, i])

            # Store coefficients (a, b) where sigmoid(a*logit + b) gives calibrated prob
            a = lr.coef_[0][0]
            b = lr.intercept_[0]
            self.coefficients.append((a, b))

    def transform(self, logits):
        """
        Transform the logits using the fitted Platt scaling parameters
        
        Args:
            logits: Model predictions (before sigmoid) - shape (n_samples, n_classes)
        
        Returns:
            Calibrated probabilities - shape (n_samples, n_classes)
        """
        calibrated_probs = np.zeros((logits.shape[0], self.num_classes))

        for i in range(self.num_classes):
            if len(self.coefficients) > i:
                a, b = self.coefficients[i]
                # Apply Platt scaling: P(y=1) = sigmoid(a*logit + b)
                calibrated_logits = a * logits[:, i] + b
                calibrated_probs[:, i] = 1 / (1 + np.exp(-calibrated_logits))
            else:
                # If no coefficients fitted, use sigmoid
                calibrated_probs[:, i] = 1 / (1 + np.exp(-logits[:, i]))

        return calibrated_probs

    def save(self, path):
        """Save the fitted coefficients"""
        np.save(path, self.coefficients)

    def load(self, path):
        """Load the fitted coefficients"""
        self.coefficients = np.load(path, allow_pickle=True).tolist()

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
        

# Evaluation function
def evaluate_model(model, data_loader, thresholds=None, calibrator=None):
    """
    Evaluate model with different thresholds
    
    Args:
        model: PyTorch model
        data_loader: DataLoader for test set
        thresholds: List of thresholds per class or None to use 0.5
        calibrator: Optional PlattScaler for calibration
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_logits = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(DEVICE)
            logits = model(inputs)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Concatenate all batches
    all_logits = np.vstack(all_logits)
    all_labels = np.vstack(all_labels)

    # Apply calibration if provided
    if calibrator is not None:
        all_probs = calibrator.transform(all_logits)
    else:
        all_probs = 1 / (1 + np.exp(-all_logits))

    # Use default threshold of 0.5 if none provided
    if thresholds is None:
        thresholds = [0.5] * NUM_CLASSES

    # Convert probabilities to predictions using thresholds
    all_preds = np.zeros_like(all_probs)
    for i in range(NUM_CLASSES):
        all_preds[:, i] = (all_probs[:, i] >= thresholds[i]).astype(int)

    # Calculate metrics per class
    results = {
        'per_class': [],
        'macro': {}
    }

    f1_scores = []
    precision_scores = []
    recall_scores = []
    auroc_scores = []

    for i in range(NUM_CLASSES):
        class_metrics = {}
        class_metrics['name'] = PATHOLOGY_NAMES[i]

        # Calculate metrics
        class_metrics['f1'] = f1_score(all_labels[:, i], all_preds[:, i])
        class_metrics['precision'] = precision_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        class_metrics['recall'] = recall_score(all_labels[:, i], all_preds[:, i])
        class_metrics['auroc'] = roc_auc_score(all_labels[:, i], all_probs[:, i])

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(all_labels[:, i], all_preds[:, i]).ravel()
        class_metrics['confusion_matrix'] = {
            'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
        }

        # Store metrics for macro averaging
        f1_scores.append(class_metrics['f1'])
        precision_scores.append(class_metrics['precision'])
        recall_scores.append(class_metrics['recall'])
        auroc_scores.append(class_metrics['auroc'])

        results['per_class'].append(class_metrics)

    # Calculate macro averages
    results['macro']['f1'] = np.mean(f1_scores)
    results['macro']['precision'] = np.mean(precision_scores)
    results['macro']['recall'] = np.mean(recall_scores)
    results['macro']['auroc'] = np.mean(auroc_scores)

    return results, all_logits, all_probs, all_labels, all_preds

# Function to visualize calibration
def plot_calibration_curves(true_labels, pred_probs, calibrated_probs, save_path=None):
    """
    Plot calibration curves before and after Platt scaling
    
    Args:
        true_labels: Ground truth binary labels
        pred_probs: Original model probabilities
        calibrated_probs: Calibrated probabilities
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(3, 6, figsize=(24, 12))
    axes = axes.flatten()

    for i in range(NUM_CLASSES):
        ax = axes[i]

        # Get data for this class
        y_true = true_labels[:, i]

        # Create bins for original probabilities
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Original probabilities
        y_prob = pred_probs[:, i]
        binned_y_true = np.zeros_like(bin_centers)
        binned_counts = np.zeros_like(bin_centers)

        for j, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:])):
            in_bin = (y_prob >= bin_start) & (y_prob < bin_end)
            if np.sum(in_bin) > 0:
                binned_y_true[j] = np.mean(y_true[in_bin])
                binned_counts[j] = np.sum(in_bin)

        # Calibrated probabilities
        cal_y_prob = calibrated_probs[:, i]
        cal_binned_y_true = np.zeros_like(bin_centers)
        cal_binned_counts = np.zeros_like(bin_centers)

        for j, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:])):
            in_bin = (cal_y_prob >= bin_start) & (cal_y_prob < bin_end)
            if np.sum(in_bin) > 0:
                cal_binned_y_true[j] = np.mean(y_true[in_bin])
                cal_binned_counts[j] = np.sum(in_bin)

        # Plot original
        ax.plot(bin_centers, binned_y_true, 'o-', label='Original', color='blue')

        # Plot calibrated
        ax.plot(bin_centers, cal_binned_y_true, 'o-', label='Calibrated', color='red')

        # Plot perfect calibration
        ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')

        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('Actual probability')
        ax.set_title(f'{PATHOLOGY_NAMES[i]}')
        ax.legend()

        # Display sample counts near the points
        for j, (x, y, count) in enumerate(zip(bin_centers, binned_y_true, binned_counts)):
            if count > 0:
                ax.annotate(f'{int(count)}', (x, y), textcoords="offset points", 
                            xytext=(0,10), ha='center', fontsize=8, color='blue')

        for j, (x, y, count) in enumerate(zip(bin_centers, cal_binned_y_true, cal_binned_counts)):
            if count > 0:
                ax.annotate(f'{int(count)}', (x, y), textcoords="offset points", 
                            xytext=(0,-10), ha='center', fontsize=8, color='red')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Function to plot confusion matrices
def plot_confusion_matrices(true_labels, predictions, thresholds, label_columns, save_path):
    """
    Plot confusion matrices for each class
    
    Args:
        true_labels: Ground truth labels (n_samples, n_classes)
        predictions: Predicted probabilities (n_samples, n_classes)
        thresholds: Thresholds to apply (n_classes)
        label_columns: Names of the classes
        save_path: Path to save the plot
    """
    import seaborn as sns

    n_classes = true_labels.shape[1]
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.ravel()

    for i in range(n_classes):
        true_binary = true_labels[:, i]
        pred_binary = (predictions[:, i] >= thresholds[i]).astype(int)

        cm = confusion_matrix(true_binary, pred_binary)

        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
        axes[i].set_title(f'{label_columns[i]}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')

    for i in range(n_classes, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Function to calculate per-class metrics
def calculate_metrics(true_labels, predictions, thresholds, label_columns):
    """
    Calculate precision, recall, F1, and AUROC for each class
    
    Args:
        true_labels: Ground truth labels (n_samples, n_classes)
        predictions: Predicted probabilities (n_samples, n_classes)
        thresholds: Thresholds to apply (n_classes)
        label_columns: Names of the classes
        
    Returns:
        DataFrame with metrics for each class
    """
    n_classes = true_labels.shape[1]
    metrics = []

    for i in range(n_classes):
        true_binary = true_labels[:, i]
        pred_proba = predictions[:, i]
        pred_binary = (pred_proba >= thresholds[i]).astype(int)

        precision = precision_score(true_binary, pred_binary, zero_division=0)
        recall = recall_score(true_binary, pred_binary)
        f1 = f1_score(true_binary, pred_binary)
        auroc = roc_auc_score(true_binary, pred_proba)

        metrics.append({
            'Class': label_columns[i],
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'AUROC': auroc,
            'Threshold': thresholds[i]
        })

    # Add macro average
    metrics.append({
        'Class': 'MACRO_AVG',
        'Precision': np.mean([m['Precision'] for m in metrics]),
        'Recall': np.mean([m['Recall'] for m in metrics]),
        'F1': np.mean([m['F1'] for m in metrics]),
        'AUROC': np.mean([m['AUROC'] for m in metrics]),
        'Threshold': np.mean(thresholds)
    })

    return pd.DataFrame(metrics)

def main():
    # Load configuration and paths
    config = {
        'resize_dim': 400,  # Match your model's expected input size
        'num_slices': 100,
        'slice_sampling': 'center',
    }

    # Data directories
    dicom_dir = "/labs/khatrilab/maestri/classes/cs231n/data/ctrate/data_volumes/dataset/"
    label_file = "/labs/khatrilab/maestri/classes/cs231n/data/ctrate/train_predicted_labels.csv"
    metadata_file = "/labs/khatrilab/maestri/classes/cs231n/data/ctrate/"
    model_name = "vlm_challenge"
    output_dir = "/labs/khatrilab/maestri/classes/cs231n/figures/"+model_name+"/"
    train_csv = "/labs/khatrilab/maestri/classes/cs231n/data/ctrate/train_predicted_labels.csv"
    val_csv = "/labs/khatrilab/maestri/classes/cs231n/data/ctrate/valid_predicted_labels.csv"

    data_dirs = dicom_dir  # This should be the base directory for CT data

    # 1. Load your model
    model = build_fusion_model()
    model.eval()

    # 2. Load validation data for calibration
    val_loader = load_validation_data(
        config=config,
        data_dirs=data_dirs,
        metadata_file=metadata_file,
        val_csv=val_csv,
        batch_size=10  # Adjust based on your GPU memory
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

    # 4. Evaluate original model with custom thresholds
    print("Evaluating original model with custom thresholds...")
    val_probs = 1 / (1 + np.exp(-val_logits))
    val_preds_custom = np.zeros_like(val_probs)
    for i in range(NUM_CLASSES):
        val_preds_custom[:, i] = (val_probs[:, i] >= CUSTOM_THRESHOLDS[i]).astype(int)

    # Calculate metrics for custom thresholds
    custom_f1 = f1_score(val_labels, val_preds_custom, average='macro')
    custom_auroc = roc_auc_score(val_labels, val_probs, average='macro')
    print(f"Original model with custom thresholds - F1: {custom_f1:.4f}, AUROC: {custom_auroc:.4f}")

    # 5. Evaluate original model with 0.5 thresholds
    print("Evaluating original model with 0.5 thresholds...")
    val_preds_default = np.zeros_like(val_probs)
    for i in range(NUM_CLASSES):
        val_preds_default[:, i] = (val_probs[:, i] >= 0.5).astype(int)

    # Calculate metrics for default thresholds
    default_f1 = f1_score(val_labels, val_preds_default, average='macro')
    print(f"Original model with 0.5 thresholds - F1: {default_f1:.4f}, AUROC: {custom_auroc:.4f}")

    # 6. Split validation data for Platt scaling
    # Use 80% for fitting, 20% for evaluation
    np.random.seed(42)  # For reproducibility
    indices = np.random.permutation(len(val_logits))
    split_idx = int(len(indices) * 0.8)

    fit_indices = indices[:split_idx]
    eval_indices = indices[split_idx:]

    fit_logits = val_logits[fit_indices]
    fit_labels = val_labels[fit_indices]

    eval_logits = val_logits[eval_indices]
    eval_labels = val_labels[eval_indices]
    eval_probs = val_probs[eval_indices]

    # 7. Fit Platt scaling
    print("Fitting Platt scaling...")
    platt_scaler = PlattScaler(num_classes=NUM_CLASSES)
    platt_scaler.fit(fit_logits, fit_labels)

    # Save the scaler for future use
    platt_scaler.save('platt_scaler.joblib')

    # 8. Apply Platt scaling to evaluation set
    print("Applying Platt scaling...")
    calibrated_probs = platt_scaler.transform(eval_logits)

    # 9. Evaluate with Platt scaling and 0.5 thresholds
    print("Evaluating with Platt scaling and 0.5 thresholds...")
    cal_preds = np.zeros_like(calibrated_probs)
    for i in range(NUM_CLASSES):
        cal_preds[:, i] = (calibrated_probs[:, i] >= 0.5).astype(int)

    # Calculate metrics after calibration
    cal_f1 = f1_score(eval_labels, cal_preds, average='macro')
    cal_auroc = roc_auc_score(eval_labels, calibrated_probs, average='macro')
    print(f"Calibrated model with 0.5 thresholds - F1: {cal_f1:.4f}, AUROC: {cal_auroc:.4f}")

    # Plot confusion matrices and calculate metrics (Platt scaling, 0.5 thresholds)
    print("Plotting confusion matrices with Platt scaling and 0.5 thresholds...")
    plot_confusion_matrices(
        eval_labels,
        calibrated_probs,
        [0.5] * NUM_CLASSES,
        PATHOLOGY_NAMES,
        'platt_scaling_confusion_matrices.pdf'
    )

    # Calculate detailed metrics with 0.5 threshold
    print("Calculating detailed metrics with Platt scaling and 0.5 thresholds...")
    metrics_df = calculate_metrics(
        eval_labels,
        calibrated_probs,
        [0.5] * NUM_CLASSES,
        PATHOLOGY_NAMES
    )

    # Save metrics to CSV
    metrics_df.to_csv('platt_scaled_validation_metrics.csv', index=False)
    print("Saved metrics to platt_scaled_validation_metrics.csv")

    # 10. Find optimal thresholds after calibration
    print("Finding optimal thresholds after calibration...")
    optimal_thresholds = []

    for i in range(NUM_CLASSES):
        best_f1 = 0
        best_threshold = 0.5

        # Test different thresholds to find optimal F1
        for threshold in np.arange(0.05, 0.95, 0.05):
            preds = (calibrated_probs[:, i] >= threshold).astype(int)
            f1 = f1_score(eval_labels[:, i], preds)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        optimal_thresholds.append(best_threshold)
        print(f"Class {i} ({PATHOLOGY_NAMES[i]}): Optimal threshold = {best_threshold:.2f}, F1 = {best_f1:.4f}")

    # 11. Apply optimal thresholds
    print("Evaluating with Platt scaling and optimal thresholds...")
    opt_preds = np.zeros_like(calibrated_probs)
    for i in range(NUM_CLASSES):
        opt_preds[:, i] = (calibrated_probs[:, i] >= optimal_thresholds[i]).astype(int)

    # Calculate metrics with optimal thresholds
    opt_f1 = f1_score(eval_labels, opt_preds, average='macro')
    print(f"Calibrated model with optimal thresholds - F1: {opt_f1:.4f}, AUROC: {cal_auroc:.4f}")

    # Calculate and save detailed metrics with optimal thresholds
    metrics_optimal_df = calculate_metrics(
        eval_labels,
        calibrated_probs,
        optimal_thresholds,
        PATHOLOGY_NAMES
    )
    metrics_optimal_df.to_csv('platt_scaled_optimal_validation_metrics.csv', index=False)
    print("Saved optimal threshold metrics to platt_scaled_optimal_validation_metrics.csv")

    # 12. Visualize calibration curves
    print("Plotting calibration curves...")
    plot_calibration_curves(
        eval_labels, 
        eval_probs, 
        calibrated_probs, 
        save_path='calibration_curves.pdf'
    )

    # 13. Save results
    print("Saving optimal thresholds...")
    np.save('optimal_thresholds.npy', np.array(optimal_thresholds))

    print("Done!")
    
if __name__ == "__main__":
    main()

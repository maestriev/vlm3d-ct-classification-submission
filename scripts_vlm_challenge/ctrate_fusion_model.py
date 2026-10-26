"""
ctrate_fusion_model.py
----------------------
Follow-up experiment to top_current_07.25.2025.py (same training pipeline,
loss helpers and DDP setup). Not the configuration used for the submission.

Differences from the submitted model:
  * 2D branch swapped to ResNet-34 (the second initialize_fusion_model
    definition overrides the ResNet-18 one).
  * ViViT dim reduced to 256.
  * The 2D branch processes slices in chunks of 40 under autocast to reduce
    GPU memory.

Usage: edit the data/output paths in the __main__ block, then
    python ctrate_fusion_model.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import models, transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, accuracy_score, recall_score, precision_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import seaborn as sns
import nibabel as nib
from functools import partial
import glob


import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset 
from torchvision import models
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from torch.cuda.amp import GradScaler, autocast

from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import numpy as np
import numpy as np
from sklearn.model_selection import train_test_split


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from torchvision import models

from tqdm import tqdm
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    accuracy_score, recall_score
)
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
import seaborn as sns

from model_architecture_helper_functions import *
import sys
import os
# Add the ctvit directory to the Python path
sys.path.append(os.path.abspath('./ctvit'))

# einops and gradient checkpointing utilities
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.utils.checkpoint import checkpoint

from model_architecture_helper_functions import *

import math


import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification with class imbalance.
    
    Args:
        alpha: Weighting factor for rare class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: Specifies the reduction to apply to the output
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Calculate probabilities
        probs = torch.sigmoid(inputs)

        # Calculate pt (probability of true class)
        pt = torch.where(targets == 1, probs, 1 - probs)

        # Calculate alpha_t (alpha for true class)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # Calculate focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class AdaptiveFocalLoss(nn.Module):
    """
    Adaptive Focal Loss that uses different alpha values for each class
    based on class frequency.
    """
    def __init__(self, class_frequencies, gamma=2.0, reduction='mean'):
        super(AdaptiveFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        # Calculate alpha values based on class frequencies
        # Higher alpha for rare classes (lower frequency)
        total_samples = sum(class_frequencies)
        self.alpha_values = torch.tensor([
            (total_samples - freq) / total_samples for freq in class_frequencies
        ], dtype=torch.float32)

    def forward(self, inputs, targets):
        device = inputs.device
        self.alpha_values = self.alpha_values.to(device)

        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Calculate probabilities
        probs = torch.sigmoid(inputs)

        # Calculate pt (probability of true class)
        pt = torch.where(targets == 1, probs, 1 - probs)

        # Get alpha for each class
        alpha_t = torch.where(targets == 1, 
                             self.alpha_values.unsqueeze(0).expand_as(targets),
                             1 - self.alpha_values.unsqueeze(0).expand_as(targets))

        # Calculate focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class WeightedFocalLoss(nn.Module):
    """
    Weighted Focal Loss that combines class weights with focal loss.
    Good for extreme class imbalances.
    """
    def __init__(self, class_weights, alpha=0.25, gamma=2.0, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def forward(self, inputs, targets):
        device = inputs.device
        self.class_weights = self.class_weights.to(device)

        # Calculate BCE loss with class weights
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Apply class weights
        weighted_bce = bce_loss * self.class_weights.unsqueeze(0).expand_as(targets)

        # Calculate probabilities
        probs = torch.sigmoid(inputs)

        # Calculate pt (probability of true class)
        pt = torch.where(targets == 1, probs, 1 - probs)

        # Calculate alpha_t
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # Calculate focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * weighted_bce

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def calculate_class_frequencies(dataset_or_dataloader, label_columns):
    """
    Calculate class frequencies from dataset or dataloader.
    
    Args:
        dataset_or_dataloader: Dataset or DataLoader object
        label_columns: List of label column names
    
    Returns:
        List of class frequencies
    """
    if hasattr(dataset_or_dataloader, '__len__') and hasattr(dataset_or_dataloader, '__getitem__'):
        # It's a dataset
        all_labels = []
        for i in range(len(dataset_or_dataloader)):
            _, _, labels = dataset_or_dataloader[i]
            all_labels.append(labels.numpy())
        all_labels = np.vstack(all_labels)
    else:
        # It's a dataloader
        all_labels = []
        for _, _, labels in dataset_or_dataloader:
            all_labels.append(labels.numpy())
        all_labels = np.vstack(all_labels)

    # Calculate frequencies for each class
    class_frequencies = all_labels.sum(axis=0).tolist()

    print("Class frequencies:")
    for i, (col, freq) in enumerate(zip(label_columns, class_frequencies)):
        print(f"{col}: {freq} ({freq/len(all_labels)*100:.1f}%)")

    return class_frequencies

def get_recommended_loss_function(class_frequencies, total_samples):
    """
    Recommend the best loss function based on class imbalance severity.
    
    Args:
        class_frequencies: List of frequencies for each class
        total_samples: Total number of samples
    
    Returns:
        Recommended loss function and parameters
    """
    # Calculate imbalance ratio (min_freq / max_freq)
    min_freq = min(class_frequencies)
    max_freq = max(class_frequencies)
    imbalance_ratio = min_freq / max_freq

    # Calculate percentage of minority classes (< 10% of samples)
    minority_classes = sum(1 for freq in class_frequencies if freq < 0.1 * total_samples)
    minority_percentage = minority_classes / len(class_frequencies)

    print(f"Imbalance ratio: {imbalance_ratio:.3f}")
    print(f"Minority classes: {minority_classes}/{len(class_frequencies)} ({minority_percentage*100:.1f}%)")

    if imbalance_ratio < 0.1 and minority_percentage > 0.3:
        # Severe imbalance - use WeightedFocalLoss
        class_weights = [(total_samples - freq) / freq for freq in class_frequencies]
        loss_fn = WeightedFocalLoss(class_weights, alpha=0.25, gamma=2.0)
        print("Recommended: WeightedFocalLoss (severe imbalance)")
    elif imbalance_ratio < 0.3:
        # Moderate imbalance - use AdaptiveFocalLoss
        loss_fn = AdaptiveFocalLoss(class_frequencies, gamma=2.0)
        print("Recommended: AdaptiveFocalLoss (moderate imbalance)")
    else:
        # Mild imbalance - use standard FocalLoss
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        print("Recommended: FocalLoss (mild imbalance)")

    return loss_fn

def setup_focal_loss(train_dataset, label_columns):
    """
    Setup the appropriate focal loss for your dataset.
    """
    # Calculate class frequencies
    class_frequencies = calculate_class_frequencies(train_dataset, label_columns)
    total_samples = len(train_dataset)

    # Get recommended loss function
    loss_fn = get_recommended_loss_function(class_frequencies, total_samples)

    return loss_fn


import torch
import numpy as np
import pandas as pd

def calculate_class_frequencies_from_dataset(dataset, label_columns):
    print("Calculating class frequencies from full dataset (no image loading)...")

    label_df = dataset.get_all_labels()
    label_columns = dataset.get_label_columns()

    label_array = label_df.drop(columns=['volume_name']).values  # shape: (N, C)
    class_frequencies = label_array.sum(axis=0)  # shape: (C,)

    print("\nClass Frequencies:")
    for col, freq in zip(label_columns, class_frequencies):
        percentage = freq / len(label_array) * 100
        print(f"{col}: {int(freq)}/{len(label_array)} ({percentage:.1f}%)")

    # Free memory
    del label_df, label_array
    gc.collect()

    return class_frequencies


def setup_focal_loss_for_training(train_dataset, label_columns, rank=0):
    """
    Setup focal loss based on dataset characteristics
    """
    # Calculate class frequencies
    class_frequencies = calculate_class_frequencies_from_dataset(train_dataset, label_columns)

    # Determine severity of imbalance
    total_samples = len(class_frequencies)
    min_freq = min(class_frequencies)
    max_freq = max(class_frequencies)
    imbalance_ratio = min_freq / max_freq

    # Count minority classes (< 5% of samples)
    minority_threshold = 0.05 * sum(class_frequencies)
    minority_classes = sum(1 for freq in class_frequencies if freq < minority_threshold)

    if rank == 0:
        print(f"Imbalance analysis:")
        print(f"  - Imbalance ratio: {imbalance_ratio:.3f}")
        print(f"  - Minority classes: {minority_classes}/{len(class_frequencies)}")
        print(f"  - Most frequent class: {max_freq} samples")
        print(f"  - Least frequent class: {min_freq} samples")

    # Choose loss function
    if imbalance_ratio < 0.05 or minority_classes > len(class_frequencies) * 0.3:

        # Severe imbalance
        class_weights = [max_freq / max(freq, 1) for freq in class_frequencies]
        criterion = WeightedFocalLoss(class_weights, alpha=0.25, gamma=2.5)
        loss_name = "WeightedFocalLoss"
    elif imbalance_ratio < 0.2:
        # Moderate imbalance
        criterion = AdaptiveFocalLoss(class_frequencies, gamma=2.0)
        loss_name = "AdaptiveFocalLoss"
    else:
        # Mild imbalance
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        loss_name = "FocalLoss"

    if rank == 0:
        print(f"Selected loss function: {loss_name}")

    return criterion


import gc
import time

def print_gpu_memory_usage():
    """Print detailed GPU memory usage statistics"""
    print("\n--- GPU MEMORY USAGE ---")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        print(f"  Reserved memory: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
        print(f"  Allocated memory: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
        print(f"  Free memory: {torch.cuda.get_device_properties(i).total_memory / 1e9 - torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
    print("------------------------\n")

def find_tensor_sizes():
    """Find the largest tensors in PyTorch's memory"""
    from collections import defaultdict

    # Get all objects tracked by PyTorch
    tensor_sizes = defaultdict(int)
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.device.type == 'cuda':
                tensor_sizes[obj.size()] += obj.element_size() * obj.nelement()
        except:
            pass

    # Print the largest tensors
    print("\n--- LARGEST TENSORS ---")
    for size, bytes in sorted(tensor_sizes.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"Size: {size}, Memory: {bytes / 1e9:.2f} GB")
    print("----------------------\n")


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))
        self.dim = dim

    def forward(self, x):
        # Make sure we're normalizing over the correct dimension
        return F.layer_norm(x, (self.dim,), self.gamma, self.beta)


import torch
import torch.nn as nn
from vit_pytorch.vivit import ViT

class SimpleVideoClassifier(nn.Module):
    def __init__(self, num_classes=18, num_frames=160):
        super().__init__()

        # initialize ViViT with parameters similar to SmallViT
        self.vivit = ViT(
            image_size = 400,
            frames = num_frames,       # number of frames (slices)
            image_patch_size = 16,
            frame_patch_size = 10,      # Process X frames at once
            num_classes = num_classes,
            dim = 256,
            spatial_depth = 6,         # spatial transformer depth
            temporal_depth = 1,        # temporal transformer depth
            heads = 8,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1,
            variant = 'factorized_encoder',
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


# ResNet-34 variant (redefines initialize_fusion_model above, so this is the one that is used)

from torchvision.models import resnet34, ResNet34_Weights
import torch.nn as nn
import torch

def initialize_fusion_model(num_classes=18, pretrained_2d=False):
    # Initialize ResNet34
    base_2d_model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained_2d else None)

    # Modify first layer for single-channel input
    base_2d_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Initialize ViViT-based video classifier
    base_video_model = SimpleVideoClassifier(num_classes=num_classes, num_frames=160)

    # Create fusion model
    fusion_model = EnhancedFusionModel(base_2d_model, base_video_model, num_classes)

    return fusion_model


class EnhancedFusionModel(nn.Module):
    def __init__(self, base_2d_model, base_video_model, num_classes=18):
        super(EnhancedFusionModel, self).__init__()

        # Store the base models
        self.base_2d_model = base_2d_model
        self.base_video_model = base_video_model

        # For ResNet34, feature dimension is 512
        self.features_2d = 512

        # Replace the classifier with Identity
        self.base_2d_model.fc = nn.Identity()

        # For the ViViT model in SimpleVideoClassifier
        # Get the features dimension from the video model
        self.features_video = self.base_video_model.vivit.mlp_head.in_features

        # Save the original video classifier and replace with Identity
        self.original_video_classifier = self.base_video_model.vivit.mlp_head
        self.base_video_model.vivit.mlp_head = nn.Identity()

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

        # Extract features from the video model
        video_features = self.base_video_model.vivit(x_video)  # [batch_size, features_video]

        # Project video features to match 2D feature dimension
        video_features = self.projection_video(video_features)

        # Process with 2D model (slice by slice) - MEMORY EFFICIENT VERSION
        x_2d = x.permute(0, 2, 1, 3, 4)  # [batch_size, depth, channels, height, width]

        # Process slices in chunks to reduce GPU memory
        chunk_size = 40  # slices per chunk
        features_2d_chunks = []

        for i in range(0, depth, chunk_size):
            end_idx = min(i + chunk_size, depth)
            chunk = x_2d[:, i:end_idx].reshape(batch_size * (end_idx - i), channels, height, width)

            # Process chunk with mixed precision to save memory
            with torch.cuda.amp.autocast(enabled=True):
                chunk_features = self.base_2d_model(chunk)

            # Reshape back
            chunk_features = chunk_features.view(batch_size, end_idx - i, -1)
            features_2d_chunks.append(chunk_features)

            # Explicitly free memory
            torch.cuda.empty_cache()

        # Concatenate all chunks
        features_2d = torch.cat(features_2d_chunks, dim=1)

        # Aggregate across slices
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

####################################################################
# extra distributed training - ddp functions

import torch.distributed as dist
import datetime

def setup(rank, world_size):
    """
    Initialize the distributed environment.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # init process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size,timeout=datetime.timedelta(minutes=30))
    torch.cuda.set_device(rank)

def cleanup():
    """
    Clean up distributed environment.
    """
    dist.destroy_process_group()
    
def train_evaluate_model_ddp(rank, world_size, config, data_dirs, label_file, metadata_file, train_csv, val_csv, output_dir, max_samples=4000, model_name="basemodel"):
    """
    Train and evaluate the CT classification model using DDP, focusing only on training and validation.
    """
    # initialize distributed environment only if world_size > 1
    if world_size > 1:
        setup(rank, world_size)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create output directory if it doesn't exist (only on master process)
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    # check if mixed precision is possible
    use_mixed_precision = hasattr(torch.cuda, 'amp')
    if use_mixed_precision:
        print(f"Rank {rank}: Using mixed precision training")
        scaler = GradScaler()
    else:
        print(f"Rank {rank}: Mixed precision training not available, using full precision")
        scaler = None

    # set hyperparameters
    batch_size = config.get('batch_size', 4)
    learning_rate = config.get('lr', 0.001)
    num_epochs = config.get('epochs', 10)
    resize_dim = config.get('resize_dim', 224)
    aggregation_method = config.get('aggregation', 'mean')
    num_slices = config.get('num_slices', 100)
    slice_sampling_method = config.get('slice_sampling', 'center')

    # Load data using train and validation CSV files
    if train_csv and os.path.exists(train_csv):
        print(f"Using train and validation splits from {train_csv} and {val_csv}")

        # Create separate datasets for train and validation
        train_dataset = CTReportDataset(
            data_folder=data_dirs+"train/",
            csv_file=metadata_file+"train_reports.csv",  # Use train metadata CSV
            label_file=train_csv,  
            resize_dim=resize_dim, 
            num_slices=num_slices, 
            slice_sampling=slice_sampling_method  
        )

        val_dataset = CTReportDataset(
            data_folder=data_dirs+"valid/",
            csv_file=metadata_file+"validation_reports.csv",       # Use validation metadata CSV
            label_file=val_csv,                                    #Use validation labels CSV
            resize_dim=resize_dim, 
            num_slices=num_slices, 
            slice_sampling=slice_sampling_method  
        )

        # print dataset information (only on master process)
        if rank == 0:
            print(f"Dataset created with {len(train_dataset)} samples")
            label_columns = train_dataset.get_label_columns()
            print(f"Available labels ({len(label_columns)}): {label_columns}")
        else:
            label_columns = train_dataset.get_label_columns()

        # Then create DataLoaders without using the split function
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=8,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=8,
            pin_memory=True
        )

    else:
        print("Creating new dataset splits")

        # limit to max samples if needed
        if max_samples < len(dataset):
            # indices for the subset
            indices = torch.randperm(len(dataset))[:max_samples]
            dataset = Subset(dataset, indices)
            print(f"Using subset of {len(dataset)} samples")

            # perform stratified train/val split (no test split)
            train_indices, val_indices = stratified_multilabel_split(
                dataset, 
                train_size=0.8, 
                val_size=0.2, 
                test_size=0,  # No test set
                output_dir=output_dir
            )

        # create data loaders with distributed samplers
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        # create distributed samplers
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        # create data loaders with distributed samplers
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            sampler=train_sampler,
            num_workers=8)

        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            sampler=val_sampler,
            num_workers=8)

    # initialize model
    model = initialize_fusion_model(num_classes=len(label_columns), pretrained_2d=True)

    # init with the slice aggregator approach

    model = model.to(device)

    # wrap the model with DistributedDataParallel

    # find_unused_parameters=True is required for the ViViT branch
    model = DDP(model, device_ids=[rank],find_unused_parameters=True)

    # Loss function and optimizer (standard BCE)
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    # training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_metrics_history = [] 
    val_metrics_history = []

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)

        # training phase
        model.train()
        train_loss = 0.0
        # tracking metrics on the training set
        all_train_preds = []
        all_train_labels = []

        for batch_idx, (videos, reports, labels) in enumerate(train_loader):
            videos = videos.to(device).float()
            labels = labels.to(device).float()

            # forward pass with autocast for mixed precision
            if use_mixed_precision:
                with autocast():
                    outputs = model(videos)
                    loss = criterion(outputs, labels)

                # backward and optimize with gradient scaling for mixed precision
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # standard full precision training
                outputs = model(videos)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

            train_loss += loss.item()

            # collect training predictions for metrics calculation
            with torch.no_grad():
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_train_preds.append(probs)
                all_train_labels.append(labels.cpu().numpy())

            # print progress (only on master process)
            if rank == 0 and (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        # average the loss across all processes
        train_loss = train_loss / len(train_loader)
        dist.all_reduce(torch.tensor(train_loss).to(device), op=dist.ReduceOp.SUM)
        train_loss = train_loss / world_size

        if rank == 0:
            train_losses.append(train_loss)

        # validation phase
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for videos, reports, labels in val_loader:
                videos = videos.to(device).float()
                labels = labels.to(device).float()

                # use autocast for mixed precision validation as well
                if use_mixed_precision:
                    with autocast():
                        outputs = model(videos)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(videos)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()

                # store predictions and labels for metrics calculation
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_val_preds.append(probs)
                all_val_labels.append(labels.cpu().numpy())

        # average the validation loss across all processes
        val_loss = val_loss / len(val_loader)
        dist.all_reduce(torch.tensor(val_loss).to(device), op=dist.ReduceOp.SUM)
        val_loss = val_loss / world_size

        if rank == 0:
            val_losses.append(val_loss)

        # calculate metrics on rank 0 to avoid duplicate work
        if rank == 0:
            # calculate all training metrics
            all_train_preds = np.vstack(all_train_preds)
            all_train_labels = np.vstack(all_train_labels)
            train_metrics = calculate_metrics(all_train_labels, all_train_preds,label_columns=label_columns)
            optimal_thresholds = train_metrics['thresholds']

            train_metrics['epoch'] = epoch + 1
            train_metrics['train_loss'] = train_loss
            train_metrics_history.append(train_metrics)

            # calculate all validation metrics
            all_val_preds = np.vstack(all_val_preds)
            all_val_labels = np.vstack(all_val_labels)
            val_metrics = calculate_metrics(all_val_labels, all_val_preds, thresholds=optimal_thresholds, label_columns=label_columns)
            val_metrics['epoch'] = epoch + 1
            val_metrics['val_loss'] = val_loss
            val_metrics_history.append(val_metrics)

            # print key metrics
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Train F1: {train_metrics['f1_macro']:.4f}, Val F1: {val_metrics['f1_macro']:.4f}")
            print(f"Train AUROC: {train_metrics['auroc']:.4f}, Val AUROC: {val_metrics['auroc']:.4f}")

            # confusion matrices
            plot_confusion_matrices(all_train_labels, all_train_preds, optimal_thresholds, label_columns, 'train', output_dir)
            plot_confusion_matrices(all_val_labels, all_val_preds, optimal_thresholds, label_columns, 'validation', output_dir)

            # learning rate scheduler step
            scheduler.step(val_loss)

            # save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.module.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                print(f"Saved best model with validation loss: {best_val_loss:.4f}")
                optimal_threshold_best = optimal_thresholds

        # synchronize all processes before next epoch
        dist.barrier()

    # final evaluation on validation set
    if rank == 0:
        print("\nFinal evaluation on validation set...")

    # load best model (all processes)
    if rank == 0:
        # Load best model on master process
        model.module.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))

    # broadcast model parameters from rank 0 to all processes
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    model.eval()
    final_val_loss = 0.0
    all_final_val_preds = []
    all_final_val_labels = []

    with torch.no_grad():
        for videos, reports, labels in val_loader:
            videos = videos.to(device).float()
            labels = labels.to(device).float()

            if use_mixed_precision:
                with autocast():
                    outputs = model(videos)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(videos)
                loss = criterion(outputs, labels)

            final_val_loss += loss.item()

            probs = torch.sigmoid(outputs).cpu().numpy()
            all_final_val_preds.append(probs)
            all_final_val_labels.append(labels.cpu().numpy())

    # average the final validation loss across all processes
    final_val_loss = final_val_loss / len(val_loader)
    dist.all_reduce(torch.tensor(final_val_loss).to(device), op=dist.ReduceOp.SUM)
    final_val_loss = final_val_loss / world_size

    # only calculate metrics on rank 0
    if rank == 0:
        all_final_val_preds = np.vstack(all_final_val_preds)
        all_final_val_labels = np.vstack(all_final_val_labels)

        # calculate final validation metrics
        final_val_metrics = calculate_metrics(all_final_val_labels, all_final_val_preds, thresholds=optimal_threshold_best,label_columns=label_columns)
        final_val_metrics['val_loss'] = final_val_loss

        # final threshold analysis
        threshold_dir = os.path.join(output_dir, 'final_threshold_analysis')
        os.makedirs(threshold_dir, exist_ok=True)

        # Visualization of prediction distributions and thresholds
        for i, class_name in enumerate(label_columns):
            plt.figure(figsize=(10, 5))

            # Get predictions and true labels for this class
            preds = all_final_val_preds[:, i]
            labels = all_final_val_labels[:, i]

            # Plot histograms
            plt.hist(preds[labels==1], bins=50, alpha=0.5, label='True Positives')
            plt.hist(preds[labels==0], bins=50, alpha=0.5, label='True Negatives')

            plt.axvline(x=optimal_threshold_best[i], color='r', linestyle='--', 
                        label=f'Threshold: {optimal_threshold_best[i]:.3f}')

            plt.title(f'Prediction Distribution for {class_name}')
            plt.xlabel('Predicted Probability')
            plt.ylabel('Count')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(threshold_dir, f'{class_name}.pdf'))
            plt.close()

        # confusion matrix
        plot_confusion_matrices(all_final_val_labels, all_final_val_preds, optimal_threshold_best, label_columns, 'final_validation', output_dir)

        # print final validation metrics
        print(f"\nFinal Validation Results:")
        print(f"Loss: {final_val_loss:.4f}")
        print(f"AUROC: {final_val_metrics['auroc']:.4f}")
        print(f"AP: {final_val_metrics['ap']:.4f}")
        print(f"F1 Score (Macro): {final_val_metrics['f1_macro']:.4f}")
        print(f"Recall (Macro): {final_val_metrics['recall_macro']:.4f}")
        print(f"Sample Accuracy: {final_val_metrics['sample_accuracy']:.4f}")

        # plot training and validation loss
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
        plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.savefig(os.path.join(output_dir, 'loss_plot.pdf'))

        # plot training and validation metrics together
        plt.figure(figsize=(15, 10))
        metrics_to_plot = ['auroc', 'ap', 'f1_macro', 'recall_macro', 'sample_accuracy']

        for i, metric in enumerate(metrics_to_plot, 1):
            plt.subplot(2, 3, i)

            # plot training metrics
            train_values = [m[metric] for m in train_metrics_history]
            plt.plot(range(1, num_epochs + 1), train_values, label='Training', color='blue')

            # plot validation metrics
            val_values = [m[metric] for m in val_metrics_history]
            plt.plot(range(1, num_epochs + 1), val_values, label='Validation', color='orange')

            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.title(f'{metric.replace("_", " ").title()}')
            plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_val_metrics_plot.pdf'))

        # training metrics
        train_metrics_df = pd.DataFrame(train_metrics_history)
        train_metrics_df['model'] = model_name
        train_metrics_df['set'] = 'train'
        train_metrics_df.to_csv(os.path.join(output_dir, 'training_metrics.csv'), index=False)

        # validation metrics
        val_metrics_df = pd.DataFrame(val_metrics_history)
        val_metrics_df['model'] = model_name
        val_metrics_df['set'] = 'validation'
        val_metrics_df.to_csv(os.path.join(output_dir, 'validation_metrics.csv'), index=False)

        # save final validation metrics
        final_val_df = pd.DataFrame({k: [v] if not isinstance(v, list) else [str(v)] 
                           for k, v in final_val_metrics.items()})
        final_val_df['model'] = model_name
        final_val_df['set'] = 'final_validation'
        final_val_df.to_csv(os.path.join(output_dir, 'final_validation_metrics.csv'), index=False)

        # save loss values
        loss_df = pd.DataFrame({
            'model': model_name,
            'epoch': range(1, num_epochs + 1),
            'train_loss': train_losses,
            'val_loss': val_losses
        })
        loss_df.to_csv(os.path.join(output_dir, 'loss_history.csv'), index=False)

    # clean up distributed environment
    cleanup()

    if rank == 0:
        return {
            'val_loss': best_val_loss,
            'final_val_loss': final_val_loss,
            'final_val_auroc': final_val_metrics['auroc'],
            'final_val_ap': final_val_metrics['ap'],
            'final_val_f1': final_val_metrics['f1_macro'],
            'final_val_recall': final_val_metrics['recall_macro'],
            'final_val_accuracy': final_val_metrics['sample_accuracy']
        }
  
    
if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.freeze_support() 

    # config
    config = {
        'batch_size': 8,
        'lr': 0.0005,
        'epochs': 5,
        'resize_dim': 400,
        'num_slices': 240, 
        'slice_sampling': 'center',
        'aggregation': 'mean'
    }

    # data directories
    dicom_dir = "/labs/khatrilab/maestri/classes/cs231n/data/ctrate/data_volumes/dataset/"
    label_file = "/labs/khatrilab/maestri/classes/cs231n/data/ctrate/train_predicted_labels.csv"
    metadata_file = "/labs/khatrilab/maestri/classes/cs231n/data/ctrate/"
    model_name = "vlm_challenge"
    output_dir = "/labs/khatrilab/maestri/classes/cs231n/figures/"+model_name+"/"

    train_csv = "/labs/khatrilab/maestri/classes/cs231n/data/ctrate/train_predicted_labels.csv"
    val_csv = "/labs/khatrilab/maestri/classes/cs231n/data/ctrate/valid_predicted_labels.csv"

    # get the number of available GPUs
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs!")

    if world_size > 1:
        # use multi-GPU training
        mp.spawn(
            train_evaluate_model_ddp,
            args=(world_size, config, dicom_dir, label_file, metadata_file, train_csv, val_csv, output_dir, 3000, model_name),
            nprocs=world_size,
            join=True)
    else:
        # fallback to single GPU training
        print("Only one GPU found, falling back to single GPU training")

        results = train_evaluate_model_ddp(
            0,  # rank 0 for single GPU
            world_size, 
            config, 
            dicom_dir, 
            label_file, 
            metadata_file, 
            splits_file, 
            output_dir, 
            max_samples=3000,
            model_name=model_name
        )
        print(f"Final results: {results}")

"""
model_architecture_helper_functions.py
--------------------------------------
Shared data-loading, splitting and evaluation utilities imported by the
training and calibration scripts (`from model_architecture_helper_functions import *`).

Contents
  * resize_array: trilinear resampling of a volume to a target voxel spacing.
  * CTReportDataset: PyTorch Dataset for CT-RATE NIfTI volumes. Resamples to
    1.5 x 0.75 x 0.75 mm, clips to [-1000, 1000] HU and scales to [-1, 1],
    centre crops/pads to (resize_dim, resize_dim, 240), samples the centre
    num_slices slices and loads the 18 abnormality labels.
  * random_multilabel_split / stratified_multilabel_split: train/val/test
    splitting (the stratified version balances rare labels and saves the split
    to CSV).
  * load_datasets_from_splits / load_datasets_from_splits_ddp: build
    DataLoaders from predefined split CSVs (the _ddp version adds
    DistributedSamplers for multi-GPU training).
  * calculate_metrics: AUROC, AP, F1, recall, precision and accuracy with
    class-specific threshold strategies (F1-optimal, precision-oriented,
    recall-oriented, Youden).
  * plot_confusion_matrices: per-class confusion matrices for evaluation.

Data-loading code is adapted from CT-RATE / CT-CLIP / CT2Rep (links below).
"""

####################################################################
base_folder = '/labs/khatrilab/maestri/classes/cs231n/'
####################################################################

####################################################################
batch_size = 8
####################################################################

import pydicom
import numpy as np
import os
import gdcm
import matplotlib.pyplot as plt
import scipy.ndimage
from PIL import Image
import torchvision.transforms as T
import os
import glob
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from functools import partial
import torch.nn.functional as F
import nibabel as nib
from tqdm import tqdm
from torch.utils.data import DataLoader

# Note: functions for loading data were modified from CT-RATE 
# for additional functionality, please see here for some of the 
# base functions which were adapted
# https://github.com/ibrahimethemhamamci/CT2Rep/tree/main/ctvit
# https://github.com/ibrahimethemhamamci/CT-CLIP/tree/main/scripts


####################################################################
def resize_array(array, current_spacing, target_spacing):
    """
    Resize the array to match the target spacing.

    Args:
    array (torch.Tensor): Input array to be resized.
    current_spacing (tuple): Current voxel spacing (z_spacing, xy_spacing, xy_spacing).
    target_spacing (tuple): Target voxel spacing (target_z_spacing, target_x_spacing, target_y_spacing).

    Returns:
    np.ndarray: Resized array.
    """
    # calculate new dimensions
    original_shape = array.shape[2:]

    # check for NaN or zero values in spacing - return None to skip this sample (added)
    for i in range(len(current_spacing)):
        if np.isnan(current_spacing[i]) or np.isnan(target_spacing[i]) or target_spacing[i] == 0:
            return None

    scaling_factors = [current_spacing[i] / target_spacing[i] for i in range(len(original_shape))]
    new_shape = [int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))]
    # resize array (trilinear interpolation)
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array

####################################################################
# added resizing (cropping) and slice sampling to allow for easier testing with different experiments
class CTReportDataset(Dataset):
    def __init__(self, data_folder, csv_file, label_file=None, min_slices=20, resize_dim=500, force_num_frames=True,num_slices=32, slice_sampling='center'):
        self.data_folder = data_folder
        self.csv_file = csv_file
        self.min_slices = min_slices
        self.accession_to_text = self.load_accession_text(csv_file)
        self.resize_dim = resize_dim

        # to subset the slices
        self.num_slices = num_slices
        self.slice_sampling = slice_sampling

        # load class labels if provided
        self.has_labels = label_file is not None
        if self.has_labels:
            self.label_df = pd.read_csv(label_file)
            # Convert column names to list for easier access, skip VolumeName column
            self.label_columns = list(self.label_df.columns[1:])

        self.paths = []
        self.samples = self.prepare_samples()

        self.transform = transforms.Compose([
            transforms.Resize((resize_dim, resize_dim)),
            transforms.ToTensor()])
        self.nii_to_tensor = partial(self.nii_img_to_tensor, transform=self.transform)

    def load_accession_text(self, csv_file):
        df = pd.read_csv(csv_file)
        accession_to_text = {}
        for index, row in df.iterrows():
            # this was wrong in the original CT-RATE dataset --> they updated to 'VolumeName' 
            accession_to_text[row['VolumeName']] = row["Findings_EN"], row['Impressions_EN']

        return accession_to_text

    def prepare_samples(self):
        samples = []
        for patient_folder in tqdm(glob.glob(os.path.join(self.data_folder, '*'))):
            for accession_folder in glob.glob(os.path.join(patient_folder, '*')):
                for nii_file in glob.glob(os.path.join(accession_folder, '*.nii.gz')):
                    accession_number = nii_file.split("/")[-1]

                    if accession_number not in self.accession_to_text:
                        continue

                    impression_text = self.accession_to_text[accession_number]

                    if impression_text == "Not given.":
                        impression_text = ""

                    input_text_concat = ""
                    for text in impression_text:
                        input_text_concat = input_text_concat + str(text)
                    input_text_concat = impression_text[0]
                    input_text = f'{impression_text}'

                    # if we have labels, check if this volume has labels
                    if self.has_labels and accession_number in self.label_df['VolumeName'].values:
                        samples.append((nii_file, input_text_concat, accession_number))
                    else:
                        samples.append((nii_file, input_text_concat))

                    self.paths.append(nii_file)
        return samples

    def __len__(self):
        return len(self.samples)

    def nii_img_to_tensor(self, path, transform):
        nii_img = nib.load(str(path))
        img_data = nii_img.get_fdata()

        train_df = pd.read_csv("/labs/khatrilab/maestri/classes/cs231n/data/ctrate/train_metadata.csv")
        val_df = pd.read_csv("/labs/khatrilab/maestri/classes/cs231n/data/ctrate/validation_metadata.csv")
        df = pd.concat([train_df, val_df], ignore_index=True)

        file_name = path.split("/")[-1]
        row = df[df['VolumeName'] == file_name]
        slope = float(row["RescaleSlope"].iloc[0])
        intercept = float(row["RescaleIntercept"].iloc[0])
        xy_spacing = float(row["XYSpacing"].iloc[0][1:][:-2].split(",")[0])
        z_spacing = float(row["ZSpacing"].iloc[0])

        # define the target spacing values
        target_x_spacing = 0.75
        target_y_spacing = 0.75
        target_z_spacing = 1.5

        current = (z_spacing, xy_spacing, xy_spacing)
        target = (target_z_spacing, target_x_spacing, target_y_spacing)

        img_data = img_data.transpose(2, 0, 1)

        tensor = torch.tensor(img_data)
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        img_data = resize_array(tensor, current, target)

        if img_data is None:
            print(f"Warning: Failed to resize array for {file_name}")
            return None

        img_data = img_data[0][0]
        img_data= np.transpose(img_data, (1, 2, 0))

        hu_min, hu_max = -1000, 1000
        img_data = np.clip(img_data, hu_min, hu_max)

        img_data = (((img_data ) / 1000)).astype(np.float32)
        slices=[]

        tensor = torch.tensor(img_data)
        # get the dimensions of the input tensor
        target_shape = (self.resize_dim,self.resize_dim, 240)

        # dimensions
        h, w, d = tensor.shape

        # cropping/padding values for height, width, and depth
        dh, dw, dd = target_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        # crop or pad the tensor
        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before

        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before

        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before

        tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)
        tensor = tensor.permute(2, 0, 1)
        tensor = tensor.unsqueeze(0)

        return tensor

    def get_labels(self, volume_name):
        """Get the class labels for a given volume"""
        if not self.has_labels:
            return None
        # find row with this volume name
        row = self.label_df[self.label_df['VolumeName'] == volume_name]
        if len(row) == 0:
            return torch.zeros(len(self.label_columns), dtype=torch.float32)
        # extract labels and convert to tensor
        labels = row[self.label_columns].values.astype(np.float32)
        return torch.tensor(labels[0])

    def get_label_columns(self):
        """Return the list of label column names"""
        if self.has_labels:
            return self.label_columns
        return []

    # ADDED THIS (for improved functionality)
    def get_all_labels(self):
        """
        Returns a DataFrame with 'volume_name' and all label columns.
        """
        if not self.has_labels:
            raise ValueError("Dataset does not have labels.")

        # get volume names from samples, index 2 is volume_name
        volume_names = [sample[2] for sample in self.samples]  

        # get labels for each volume
        labels = []
        for vol_name in volume_names:
            row = self.label_df[self.label_df['VolumeName'] == vol_name]
            if len(row) == 0:
                labels.append(np.zeros(len(self.label_columns), dtype=np.float32))
            else:
                labels.append(row[self.label_columns].values[0].astype(np.float32))

        # create the labeldf dataframe with the labels, volume_name = first column
        label_df = pd.DataFrame(labels, columns=self.label_columns)
        label_df.insert(0, 'volume_name', volume_names)  
        return label_df

    def __getitem__(self, index):

        max_attempts = len(self.samples)
        attempts = 0
        while attempts < max_attempts:

            if self.has_labels:
                nii_file, input_text, volume_name = self.samples[index]
                video_tensor = self.nii_to_tensor(nii_file)
                # clean text
                input_text = str(input_text)
                input_text = input_text.replace('"', '')
                input_text = input_text.replace('\'', '')
                input_text = input_text.replace('(', '')
                input_text = input_text.replace(')', '')

                # if video_tensor (volume) is None, move to the next slice
                if video_tensor is None:
                    index = (index + 1) % len(self.samples)
                    attempts += 1
                    continue

                # downsample the slices (added here to not use full volume)!
                # center slicing - take the middle of the volume!
                total_slices = video_tensor.shape[1]

                if self.num_slices < total_slices:
                    if self.slice_sampling == 'center':
                        center = total_slices // 2
                        half = self.num_slices // 2
                        start = max(0, center - half)
                        end = min(total_slices, start + self.num_slices)
                        selected_indices = np.arange(start, end)

                    video_tensor = video_tensor[:, selected_indices, :, :]

                # get labels
                labels = self.get_labels(volume_name)
                return video_tensor, input_text, labels
            else:
                nii_file, input_text = self.samples[index]
                video_tensor = self.nii_to_tensor(nii_file)
                input_text = str(input_text)
                input_text = input_text.replace('"', '')
                input_text = input_text.replace('\'', '')
                input_text = input_text.replace('(', '')
                input_text = input_text.replace(')', '')
                return video_tensor, input_text

####################################################################
# this is random multilabel splitting, only used for prelim testing         
def random_multilabel_split(dataset, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42):
    """
    Fast random train/val/test split for multilabel datasets (no stratification by class).
    
    Args:
        dataset: The dataset to split
        train_size: Proportion for training set
        val_size: Proportion for validation set
        test_size: Proportion for test set
        random_state: Random seed for reproducibility
        
    Returns:
        train_indices, val_indices, test_indices
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Splits must sum to 1"

    indices = np.arange(len(dataset))

    # first split: train vs (val + test)
    remaining_size = val_size + test_size
    first_test_size = remaining_size / (train_size + remaining_size)

    train_indices, temp_indices = train_test_split(
        indices,
        test_size=first_test_size,
        random_state=random_state,
        shuffle=True)

    # second split: val vs test
    val_ratio = val_size / remaining_size
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=1 - val_ratio,
        random_state=random_state,
        shuffle=True)
    print(f"Random split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    return train_indices, val_indices, test_indices

####################################################################
# this one balances the classes per data split (what we want!)
def stratified_multilabel_split(
    dataset, train_size=0.7, val_size=0.2, test_size=0.1, 
    random_state=42, output_dir='.'
):
    """
    Performs a stratified train/val/test split for a multi-label dataset.
    Saves:
      - volume_splits.csv: volume_name, labels, split assignment
      - split_info.csv: class-wise split statistics
    """
    import numpy as np
    import pandas as pd
    import os
    from sklearn.model_selection import train_test_split
    from collections import Counter

    orig_dataset = dataset.dataset
    full_label_df = orig_dataset.get_all_labels()
    # only include rows for the subset indices
    subset_indices = dataset.indices
    label_df = full_label_df.iloc[subset_indices].reset_index(drop=True)
    # Get all labels and volume names in one call
    volume_names = label_df['volume_name'].values
    all_labels = label_df.drop(columns='volume_name').values
    label_columns = label_df.columns[1:]

    dataset_length = len(label_df)
    indices = np.arange(dataset_length)

    # identify most imbalanced classes
    class_counts = np.sum(all_labels, axis=0)
    class_ratios = class_counts / dataset_length
    num_focus_classes = min(8, all_labels.shape[1])
    focus_class_indices = np.argsort(class_ratios)[:num_focus_classes]
    focus_labels = all_labels[:, focus_class_indices]

    # vectorized signature calculation
    powers_of_two = 2 ** np.arange(num_focus_classes)
    signatures = (focus_labels * powers_of_two).sum(axis=1)

    # stratified split
    np.random.seed(random_state)
    train_indices, temp_indices = [], []
    for sig, count in Counter(signatures).items():
        sig_indices = indices[signatures == sig]
        np.random.shuffle(sig_indices)
        n_train = max(1, int(count * train_size))
        train_indices.extend(sig_indices[:n_train])
        temp_indices.extend(sig_indices[n_train:])

    train_indices = np.array(train_indices)
    temp_indices = np.array(temp_indices)

    # validation/test split
    val_ratio = val_size / (val_size + test_size)
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=1-val_ratio, random_state=random_state)

    # save all volume names, labels, and split assignments in a single CSV
    split_assignment = np.full(dataset_length, 'unassigned', dtype=object)
    split_assignment[train_indices] = 'train'
    split_assignment[val_indices] = 'val'
    split_assignment[test_indices] = 'test'

    final_df = label_df.copy()
    final_df['split'] = split_assignment
    final_df.to_csv(os.path.join(output_dir, 'volume_splits.csv'), index=False)
    print(f"Saved all split assignments to {os.path.join(output_dir, 'volume_splits.csv')}")

    # keep the split summary info
    split_info = pd.DataFrame({
        'class': label_columns,
        'total': all_labels.sum(axis=0),
        'train': all_labels[train_indices].sum(axis=0),
        'val': all_labels[val_indices].sum(axis=0),
        'test': all_labels[test_indices].sum(axis=0)})
    split_info['train_pct'] = split_info['train'] / split_info['total'] * 100
    split_info['val_pct'] = split_info['val'] / split_info['total'] * 100
    split_info['test_pct'] = split_info['test'] / split_info['total'] * 100
    split_info.to_csv(os.path.join(output_dir, 'split_info.csv'), index=False)

    print(f"Saved class split summary to {os.path.join(output_dir, 'split_info.csv')}")

    return train_indices, val_indices, test_indices    
    
def get_all_volume_names(self):
    """
    Returns a list of all volume names in the dataset.
    """
    # extract the volume names from samples
    # samples = a list of tuples/lists where index 2 contains volume_name
    return [sample[2] for sample in self.samples]

####################################################################
# this function loads our predefined stratified splits
def load_datasets_from_splits(dataset, splits_file, batch_size=8, num_workers=8):
    """
    Load datasets using pre-defined splits from a CSV file.
    
    Args:
        dataset: The full dataset object (CTReportDataset instance)
        splits_file: Path to CSV file containing volume_name and split assignment
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        
    Returns:
        train_loader, val_loader, test_loader: DataLoaders for each split
    """
    import pandas as pd
    import os
    from torch.utils.data import DataLoader, Subset

    # check that splits file exists
    if not os.path.exists(splits_file):
        raise FileNotFoundError(f"Split file {splits_file} does not exist")
    splits_df = pd.read_csv(splits_file)

    # make sure required columns exist
    if 'volume_name' not in splits_df.columns or 'split' not in splits_df.columns:
        raise ValueError(f"Split file must contain 'volume_name' and 'split' columns")

    # get all volume names in the dataset, volume names directly from the dataset
    all_labels_df = dataset.get_all_labels()
    dataset_volume_names = all_labels_df['volume_name'].tolist()

    # indices for each split
    train_indices = []
    val_indices = []
    test_indices = []

    # map from volume name to dataset index
    volume_to_idx = {}
    for idx, sample in enumerate(dataset.samples):
        volume_name = sample[2]
        volume_to_idx[volume_name] = idx

    # assign indices based on the split
    for _, row in splits_df.iterrows():
        volume_name = row['volume_name']
        split = row['split']

        # skip volumes not in the dataset (e.g. if dataset is a subset)
        if volume_name not in volume_to_idx:
            continue

        idx = volume_to_idx[volume_name]
        # train, val, test indices
        if split == 'train':
            train_indices.append(idx)
        elif split == 'val':
            val_indices.append(idx)
        elif split == 'test':
            test_indices.append(idx)
    print(f"Split sizes: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

    # create subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


####################################################################
# this function loads our predefined stratified splits, but makes it compatible with
# our distributed training environment for multi-GPU

def load_datasets_from_splits_ddp(dataset, train_csv, valid_csv, batch_size=8, num_workers=8, rank=0, world_size=1):
    """
    Load datasets using pre-defined splits from separate train and validation CSV files with proper DistributedSampler.
    
    Args:
        dataset: The full dataset object (CTReportDataset instance)
        train_csv: Path to CSV file containing volume names for training
        valid_csv: Path to CSV file containing volume names for validation
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        rank: Process rank for distributed training
        world_size: Total number of processes
        
    Returns:
        Tuple containing:
        - train_loader, val_loader: DataLoaders for each split
        - train_sampler, val_sampler: Corresponding DistributedSamplers
    """
    import pandas as pd
    import os
    from torch.utils.data import DataLoader, Subset
    from torch.utils.data.distributed import DistributedSampler

    # Check if files exist
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Train file {train_csv} does not exist")
    if not os.path.exists(valid_csv):
        raise FileNotFoundError(f"Validation file {valid_csv} does not exist")

    # Load dataframes
    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)

    # Verify required columns exist
    if 'VolumeName' not in train_df.columns:
        raise ValueError(f"Train file must contain 'VolumeName' column")
    if 'VolumeName' not in valid_df.columns:
        raise ValueError(f"Validation file must contain 'VolumeName' column")

    # Get volume names for train and validation
    train_volumes = set(train_df['VolumeName'].tolist())
    valid_volumes = set(valid_df['VolumeName'].tolist())

    # Map from volume name to dataset index
    volume_to_idx = {}
    for idx, sample in enumerate(dataset.samples):
        volume_name = sample[2]  # Assuming volume_name is at index 2
        volume_to_idx[volume_name] = idx

    # Create indices for each split
    train_indices = []
    val_indices = []

    # Add indices based on volume names
    for volume_name, idx in volume_to_idx.items():
        if volume_name in train_volumes:
            train_indices.append(idx)
        elif volume_name in valid_volumes:
            val_indices.append(idx)

    # Print split sizes on rank 0
    if rank == 0:
        print(f"Split sizes: train={len(train_indices)}, val={len(val_indices)}")

    # Create subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # Create data loaders with distributed samplers
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # Sampler will handle shuffling
        sampler=train_sampler,
        num_workers=num_workers)

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers)

    # Return loaders and samplers
    return train_loader, val_loader, train_sampler, val_sampler


####################################################################
# metrics (for our 18 lung abnormality classifications)


from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, accuracy_score, recall_score, precision_score

def calculate_metrics(true_labels, predictions, thresholds=None, label_columns=None):
    """
    Calculate multiple classification metrics for multi-label data with class-specific threshold strategies.

    Args:
        true_labels: Ground truth labels as numpy array
        predictions: Predicted probabilities as numpy array
        thresholds: List of thresholds for converting probabilities to binary predictions.
                    If None, uses class-specific strategies to find optimal thresholds.
        label_columns: List of class names (required if thresholds=None for class-specific strategies)

    Returns:
        Dictionary of metrics
    """
    # Set up metrics dictionary
    metrics = {}

    # AUROC
    try:
        metrics['auroc'] = roc_auc_score(true_labels, predictions, average='macro')
    except ValueError:  # This can happen if a class has only one label type in the set
        metrics['auroc'] = float('nan')

    # Average precision
    try:
        metrics['ap'] = average_precision_score(true_labels, predictions, average='macro')
    except ValueError:
        metrics['ap'] = float('nan')

    # Define threshold-finding functions
    def find_optimal_f1_threshold(y_true, y_pred):
        best_f1 = 0
        best_threshold = 0.5

        for threshold in np.arange(0.01, 0.99, 0.01):
            y_pred_binary = (y_pred >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred_binary, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return best_threshold

    def find_precision_oriented_threshold(y_true, y_pred, target_precision=0.7, min_recall=0.6):
        best_f1 = 0
        best_threshold = 0.5

        for threshold in np.arange(0.01, 0.99, 0.01):
            y_pred_binary = (y_pred >= threshold).astype(int)
            recall = recall_score(y_true, y_pred_binary, zero_division=0)

            # Only consider thresholds that maintain minimum recall
            if recall >= min_recall:
                precision = precision_score(y_true, y_pred_binary, zero_division=0)
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

        return best_threshold

    def find_recall_oriented_threshold(y_true, y_pred, target_recall=0.9, min_precision=0.4):
        best_f1 = 0
        best_threshold = 0.5

        for threshold in np.arange(0.01, 0.99, 0.01):
            y_pred_binary = (y_pred >= threshold).astype(int)
            precision = precision_score(y_true, y_pred_binary, zero_division=0)

            # Only consider thresholds that maintain minimum precision
            if precision >= min_precision:
                recall = recall_score(y_true, y_pred_binary, zero_division=0)
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

        return best_threshold

    def find_youdens_threshold(y_true, y_pred):
        # Original Youden's Index approach
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        specificity = 1 - fpr
        youdens_index = tpr + specificity - 1
        optimal_idx = np.argmax(youdens_index)

        if len(thresholds) > 0 and optimal_idx < len(thresholds):
            return thresholds[optimal_idx]
        else:
            return 0.5

    # If thresholds are not provided, compute them based on class-specific strategies
    if thresholds is None:
        if label_columns is None:
            raise ValueError("label_columns must be provided when thresholds=None")

        # Define class-specific strategies based on confusion matrix analysis
        class_specific_strategies = {
            # High precision needed (currently many false positives)
            'high_precision_classes': [
                'Peribronchial thickening', 'Bronchiectasis',
                'Interlobular septal thickening', 'Hiatal hernia',
                'Mosaic attenuation pattern', 'Pericardial effusion',
                'Atelectasis','Lung nodule', 'Pulmonary fibrotic sequela'
            ],

            # Balanced approach needed
            'balanced_classes': [
                'Emphysema', 'Lymphadenopathy',
                'Cardiomegaly', 'Medical material',
                'Lung opacity'
            ],

            # Current approach working well - maintain high recall
            'high_recall_classes': [
                'Pleural effusion', 'Coronary artery wall calcification',
                'Arterial wall calcification'
            ]
        }
        class_thresholds = []

        # Find optimal thresholds for each class using the appropriate strategy
        for i, class_name in enumerate(label_columns):
            try:
                y_true = true_labels[:, i]
                y_pred = predictions[:, i]

                if class_name in class_specific_strategies['high_precision_classes']:
                    threshold = find_precision_oriented_threshold(y_true, y_pred)
                elif class_name in class_specific_strategies['high_recall_classes']:
                    threshold = find_recall_oriented_threshold(y_true, y_pred)
                else:  # balanced_classes
                    threshold = find_optimal_f1_threshold(y_true, y_pred)

                class_thresholds.append(threshold)

            except Exception as e:
                print(f"Error finding threshold for class {class_name}: {e}")
                class_thresholds.append(0.5)
    else:
        class_thresholds = thresholds

    metrics['thresholds'] = class_thresholds

    # Apply the class-specific thresholds
    binary_preds = np.zeros_like(predictions, dtype=int)
    for i in range(true_labels.shape[1]):
        binary_preds[:, i] = (predictions[:, i] >= class_thresholds[i]).astype(int)

    # Calculate metrics using binary predictions
    metrics['f1_macro'] = f1_score(true_labels, binary_preds, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(true_labels, binary_preds, average='macro', zero_division=0)
    metrics['precision_macro'] = precision_score(true_labels, binary_preds, average='macro', zero_division=0)

    # Accuracy metrics
    exact_match = np.all(binary_preds == true_labels, axis=1).mean()
    sample_accuracy = np.mean(np.sum(binary_preds == true_labels, axis=1) / true_labels.shape[1])
    metrics['sample_accuracy'] = sample_accuracy

    # Calculate per-class metrics
    per_class_f1 = []
    per_class_recall = []
    per_class_precision = []
    per_class_accuracy = []
    per_class_auc = []

    for i in range(true_labels.shape[1]):
        # F1 and Recall
        class_f1 = f1_score(true_labels[:, i], binary_preds[:, i], zero_division=0)
        class_recall = recall_score(true_labels[:, i], binary_preds[:, i], zero_division=0)
        # Precision
        class_precision = precision_score(true_labels[:, i], binary_preds[:, i], zero_division=0)
        # Accuracy
        class_accuracy = np.mean(binary_preds[:, i] == true_labels[:, i])
        # AUC per class
        try:
            class_auc = roc_auc_score(true_labels[:, i], predictions[:, i])
        except ValueError:
            class_auc = float('nan')

        per_class_f1.append(class_f1)
        per_class_recall.append(class_recall)
        per_class_precision.append(class_precision)
        per_class_accuracy.append(class_accuracy)
        per_class_auc.append(class_auc)

    # Store per-class metrics
    metrics['per_class_f1'] = per_class_f1
    metrics['per_class_recall'] = per_class_recall
    metrics['per_class_precision'] = per_class_precision
    metrics['per_class_accuracy'] = per_class_accuracy
    metrics['per_class_auc'] = per_class_auc

    return metrics


####################################################################
# confusion matrices (during eval)

import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_confusion_matrices(true_labels, predictions, thresholds, label_columns, set_name, output_dir):
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
    plt.savefig(os.path.join(output_dir, f'{set_name}_confusion_matrices.pdf'))
    plt.close()

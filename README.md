# CT Abnormality Classification — ResNet + ViViT Fusion (VLM3D Challenge)

- Multi-label classification of 18 chest CT abnormalities, trained on CT-RATE.
A 2D ResNet-18 (applied slice by slice) and a 3D ViViT are fused with learned
attention weights. Per-class probabilities are then calibrated so that a 0.5
threshold works for every class.

- Challenge: Submission to the [VLM3D Challenge](https://research.forithmus.com/collections/vlm3d-challenge)
(3D Chest CT abnormality classification task).

- Project report: [CS231N Final Project Report (Stanford, 2025)](https://cs231n.stanford.edu/2025/papers/text_file_840274985-CS231N_Final_Project_Report_EvanMaestri.pdf)

## Repository layout

```
docker/
  process.py                  Competition inference script (/input → /output/results.json)
  requirements.txt            Python dependencies for the container
  calibration_params.joblib   Per-class (scale, shift) calibration parameters
  best_model.pth              Model weights
  
scripts_vlm_challenge/
  top_current_07.25.2025.py   Training script for the submitted model
                              (ResNet-18 + ViViT dim 384, 160 slices, multi-GPU DDP)
  ctrate_fusion_model.py      Follow-up experiment (ResNet-34, ViViT dim 256, chunked 2D branch, not used in final submission)
  model_architecture_helper_functions.py
                              Dataset, data splitting, metrics and plotting utilities
  model_calibration.py        Scale/shift calibration → calibration_params.joblib (used in submission)
  platt_scaling.py            Earlier Platt-scaling calibration experiment
  plotting.R                  Figures for the report/poster
```

## Model

| Component | Details |
|---|---|
| Input | CT resampled to 1.5 × 0.75 × 0.75 mm, clipped to [-1000, 1000] HU, scaled to [-1, 1], cropped/padded to 400 × 400 × 240; only the middle 160 slices used |
| 2D branch | ResNet-18 with a single-channel first conv, features averaged over slices |
| 3D branch | ViViT (vit_pytorch, factorized encoder, dim 384, 16 × 16 × 10 patches) |
| Fusion | Attention weights over the two feature vectors → linear classifier (18 outputs) |
| Loss | BCEWithLogitsLoss, Adam (lr 5e-4), ReduceLROnPlateau, batch size 8, 5 epochs |
| Calibration | Per class: `p_cal = min(1, p * scale + shift)`, grid-searched to maximise F1 at 0.5 |

## Pipeline

1. **Train**: `python scripts_vlm_challenge/top_current_07.25.2025.py` → `best_model.pth`
2. **Calibrate**: `python scripts_vlm_challenge/model_calibration.py` → `calibration_params.joblib`
3. **Inference**: place `best_model.pth` and `calibration_params.joblib` in `/opt/app/models/`
   inside the container and run `docker/process.py`.

## Data

Trained on [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) using the
predicted multi-abnormality labels (`train_predicted_labels.csv` / `valid_predicted_labels.csv`).
Both files are in the  `ibrahimhamamci/CT-RATE` under `dataset/multi_abnormality_labels/`. 
The split is by patient: 20,000 train / 1,304 validation patients (47,149 / 3,039 volumes).

The training and calibration scripts import `model_architecture_helper_functions.py`.

Data loading was adapted from
[CT-CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP) and
[CT2Rep](https://github.com/ibrahimethemhamamci/CT2Rep), using the CT-RATE dataset.

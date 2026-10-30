# TI mmWave Radar Fall Detection Pipeline

This directory contains the complete pipeline for the **TI mmWave Radar Dataset** (102 CSV point cloud recordings collected across subjects *Areeb*, *Raffay*, and *Towsif*).

## Pipeline Features
* **Zero Subject Leakage**: 3-Fold Leave-One-Subject-Out (LOSO) cross-validation protocol.
* **Point Cloud Representations**:
  1. **Representation 1**: Micro-Doppler Spectrograms (ResNet-18)
  2. **Representation 2**: 2D Spatial Projections (ResNet-18)
  3. **Representation 3**: 3D Spatial-Temporal Point Clouds (PointNet++)
  4. **Non-AI Baseline**: Physical heuristic rules ($\Delta z$, $|v|_{\max}$, $z_{\text{final}}$)

## Structure
* `src/`: Modular Python scripts for data loading, representation generation, PyTorch models, LOSO trainer, and rule baseline.
* `notebooks/`: Jupyter notebooks 01 through 08 detailing EDA, baseline, optimization, and evaluation.
* `models/`: Trained model checkpoints and benchmark JSON results.

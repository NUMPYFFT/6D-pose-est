# 6D Pose Estimation with PointNet and ICP

This repository implements a robust 6D pose estimation pipeline using **PointNet** for coarse pose estimation and **Iterative Closest Point (ICP)** for fine refinement.

For a detailed technical analysis and experimental results, please refer to [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).

## Project Structure

```
.
├── config.py           # Centralized configuration (argparse)
├── data.py             # Dataset loading (PoseDataset)
├── model.py            # PointNet and PointNet++ architectures
├── train.py            # Training script
├── eval.py             # Evaluation script (with ICP refinement)
├── utils.py            # Utility functions (loss, visualization, geometry)
├── benchmark_utils/    # Helper scripts and object metadata
├── output_images/      # Generated visualizations
└── TECHNICAL_REPORT.md # Detailed report
```

## Installation

Ensure you have the following dependencies installed:

```bash
pip install torch torchvision numpy pandas opencv-python tqdm open3d trimesh
```

## Data Setup

The project expects the following data structure by default (configurable in `config.py`):

- Training Data: `./training_data_filtered/training_data/v2.2`
- Splits: `./training_data_filtered/training_data/splits/v2`
- Objects CSV: `benchmark_utils/objects_v1.csv`

## Usage

### Configuration

All hyperparameters and paths are defined in `config.py`. You can override them via command-line arguments.

### Training

To train the PointNet model:

```bash
python train.py --model pointnet --epochs 100 --batch_size 128
```

Key arguments:
- `--model`: `pointnet` or `pointnet2`
- `--num_points`: Number of points to sample (default: 1024)
- `--w_pm`, `--w_geo`, `--w_trans`: Loss weights for Point Matching, Geodesic, and Translation loss.

### Evaluation

To evaluate the model with ICP refinement:

```bash
python eval.py --model pointnet --icp_stages 3 --icp_threshold 0.02
```

To disable ICP and evaluate PointNet only:

```bash
python eval.py --no_icp
```

Key arguments:
- `--no_icp`: Disable ICP refinement.
- `--icp_stages`: Number of ICP stages (1-3).
- `--icp_threshold`: Coarse ICP threshold (default: 0.02).
- `--max_rot_change`: Safety check to reject large ICP rotation changes (default: 30.0 deg).

## Results

Our best configuration (PointNet + 3-Stage ICP) achieves a **78.36% pass rate** on the validation set.

| Method | Pass Rate | Avg Rot Err | Avg Trans Err |
| :--- | :--- | :--- | :--- |
| PointNet Only | 72.26% | 4.70° | 0.47 cm |
| **PointNet + ICP** | **78.36%** | 6.42° | 0.47 cm |

See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for full ablation studies.

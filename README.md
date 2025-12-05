# 6D Pose Estimation with PointNet + ICP

This project implements a robust 6D pose estimation system for rigid objects using a deep learning approach (PointNet/DGCNN) followed by geometric refinement (Iterative Closest Point).

For a detailed explanation of the mathematical formulation, algorithms, and experimental results, please refer to the [Technical Report](TECHNICAL_REPORT.md).

## Visual Results

Below is an example of the system's output on the **Test Set** (unseen data).
*   **Green Box**: Final Prediction (ICP Refined)
*   **Red Box**: Initial Prediction (PointNet)

![Test Set Result](assets/2-48-1.png)

## Performance

Evaluated on the validation set (1717 samples):

| Method | Pass Rate (< 5°, < 1cm) | Avg Rotation Error | Avg Translation Error |
| :--- | :--- | :--- | :--- |
| **PointNet (Initial)** | 91.85% | 3.70° | 0.21 cm |
| **PointNet + ICP (Final)** | **93.65%** | **2.85°** | **0.19 cm** |

## Project Structure

```
.
├── config.py           # Centralized configuration (argparse)
├── data.py             # Dataset loading (PoseDataset)
├── model.py            # PointNet and DGCNN architectures
├── train.py            # Training script
├── eval.py             # Evaluation script (Inference -> Refinement -> Metrics)
├── icp.py              # ICP logic (Refinement, Parallel Workers, Mesh Loading)
├── utils.py            # Utility functions (loss, visualization, geometry)
├── preprocess.py       # Data preprocessing script
├── models/             # Object meshes and metadata
└── TECHNICAL_REPORT.md # Detailed report
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Data Setup

The project expects the following data structure by default (configurable in `config.py`):

- Training Data: `./training_data_filtered/training_data/v2.2`
- Splits: `./training_data_filtered/training_data/splits/v2`
- Objects CSV: `models/objects_v1.csv`

## Usage

### Configuration

All hyperparameters and paths are defined in `config.py`. You can override them via command-line arguments.

### Data Preprocessing

To speed up training and evaluation, we preprocess the dataset (point cloud generation, downsampling, normal estimation) and save it to disk.

```bash
python preprocess.py --split train
python preprocess.py --split val
python preprocess.py --split test
```

This will create a `preprocessed/` folder inside your data directory containing `.npz` files for each sample.

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
python eval.py --icp_stages 3 --icp_threshold 0.02 --batch_size 1
```

To disable ICP and evaluate PointNet only:

```bash
python eval.py --no_icp --batch_size 128
```

To evaluate on the **Test Set** (generates `test_metrics_icp.csv`):

```bash
python eval.py --split test --checkpoint_path model_weights/pointnet_v2.pth
```

Key arguments:
- `--split`: Dataset split to use (`val` or `test`).
- `--no_icp`: Disable ICP refinement.
- `--icp_stages`: Number of ICP stages (1-3).
- `--icp_threshold`: Coarse ICP threshold (default: 0.02).
- `--max_rot_change`: Safety check to reject large ICP rotation changes (default: 30.0 deg).
- `--batch_size`: Batch size for inference (default: 128).
- `--device`: Device to run on (default: cuda).


# 6D Pose Estimation: PointNet + Guarded ICP

This project estimates an object's 6D camera-frame pose from RGB-D data, an object segmentation mask, and a canonical mesh. PointNet predicts a coarse pose; three-stage ICP refines it only when geometric checks show a meaningful improvement.

The current validated baseline on the corrected validation cache contains 1,705 valid object instances across 236 scenes:

| Method | Mean rotation error | Mean translation error |
|---|---:|---:|
| PointNet | 3.57° | 0.21 cm |
| PointNet + ICP | **2.73°** | **0.19 cm** |

The newly added partial-view augmentation and guarded-ICP acceptance logic still need a retraining and evaluation run before they are reported as improvements.

For methodology, results, and the experiment plan, see [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).

## Project layout

```text
config.py              Configuration and data/checkpoint paths
data.py                RGB-D dataset loading, caching, and training augmentation
model.py               Class-conditioned PointNet pose regressor
loss.py                Pose and symmetry-aware losses
icp.py                 Canonical-mesh loading and guarded three-stage ICP
train.py               Model training with progress reporting
preprocess.py          Train/validation/test point-cloud cache generation
test_inference.py      Reusable labelled and test-set inference pipelines
error_analysis.py      Per-instance, per-object, and per-scene error reports
evaluation_workflows.py Compact labelled-evaluation and ablation workflows
pose_visualization.py  2D, zoomed, and interactive 3D pose visualizations
see_data.ipynb         Clean notebook orchestration and data exploration
TECHNICAL_REPORT.md    Detailed technical report
```

## Setup

Install the dependencies into the project environment:

```bash
pip install -r requirements.txt
```

Set the training data directory, split directory, and checkpoint path in `config.py`. The train, validation, and test caches must be built from their corresponding data roots.

## Build point-cloud caches

Caching performs depth back-projection, object-mask extraction, voxel downsampling, normal estimation, and metadata preparation. Run it once per split:

```bash
python preprocess.py --split train
python preprocess.py --split val
python preprocess.py --split test
```

Cache generation shows a live `Building <split> cache` progress bar with sample rate and valid-point count. It safely resumes: existing `.npz` files are skipped and reported in the final summary.

### Optional fast training cache

The regular training cache has one compressed NPZ file per object. To reduce Windows file-open and decompression overhead, pack the completed train cache into shared memory-mapped arrays once:

```bash
python build_fast_cache.py --split train
```

This creates `preprocessed/train_mmap/`. DataLoader workers memory-map the same read-only arrays automatically, while train-time occlusion, noise, and pose augmentation remain random every epoch.

## Train PointNet

The project requires CUDA-enabled PyTorch for practical training speed. In `see_data.ipynb`, run **GPU setup (one time)** once, restart the kernel, set `INSTALL_CUDA_TORCH = False`, and rerun the cell to verify that `torch.cuda.is_available()` is `True` and the RTX 4090 is detected. Training now stops with a clear error rather than silently using CPU if CUDA is unavailable.

```bash
python train.py --epochs 400 --batch_size 128
```

Training includes jitter, partial-view cropping, small rigid pose perturbations, and targeted large orientation changes: a 90° horizontal-axis rotation with probability 0.50 plus a 180° horizontal flip with probability 0.15. These label-consistent transforms enrich side-lying and flipped poses observed in scenes such as `2-6-3` and `2-77-11`. It displays live batch-level train/validation loss bars plus an epoch-level bar with train loss, validation loss, rotation error, and translation error.

The notebook's current GPU-oriented defaults use `TRAIN_BATCH_SIZE = 256` and `TRAIN_NUM_WORKERS = 12`. The larger batch improves RTX 4090 utilization and reduces the number of batches per epoch; reduce workers to 8 if the CPU becomes slower or unstable.

The notebook runs training in the active kernel so its tqdm bars render live. Set `CHECKPOINT_PATH` once, change `RUN_TRAINING` to `True`, run **Train a new checkpoint**, and set it back to `False` after training. All following notebook evaluations use that same checkpoint path.

### Overnight fine-tuning

After a full run completes, use the notebook's **Nightly fine-tuning run** cell. It initializes a separate output checkpoint from the completed best checkpoint, resets the optimizer and cosine schedule, and trains for 400 further epochs at `1e-4` learning rate. This run includes the targeted laid-down-pose augmentation. The source checkpoint is never overwritten; the nightly output is always valid because it begins as a copy of the source model and is replaced only when validation loss improves. It also saves snapshots every 25 epochs under `model_weights/snapshots/`, allowing targeted hard-scene evaluation even when a candidate does not improve the global validation average.

Run **TensorBoard training dashboard** after training begins to inspect live loss, rotation, and translation curves. It resolves the project `runs/` directory to an absolute path and refreshes every five seconds.

## Evaluate and inspect predictions

Open `see_data.ipynb` and run the compact section beginning at **ICP**, in order:

1. **Full validation evaluation** runs PointNet and guarded ICP, then saves `outputs/val_instance_error_report.csv`.
2. **Hard validation scenes** displays GT, PointNet, and ICP overlays.
3. **Examples where ICP improves PointNet** selects meaningful improvements and shows full-scene, zoomed, and rotatable 3D box overlays.
4. **Level-2 train scenes** evaluates the curated difficult scenes without training augmentation.
5. **Test-set inference** visualizes final predictions only; test poses have no ground truth.

Color convention for ICP-improvement visualizations:

- Dark green: ground truth
- Orange-red: PointNet initialization
- Bright green: final guarded ICP pose

## Core behavior

- PointNet accepts 1,024 point features per object: XYZ, RGB, and normals.
- Rotation uses a 6D representation; translation is predicted relative to the observed-cloud centroid.
- ICP uses point-to-plane stages at 1.0 cm and 0.5 cm, followed by point-to-point ICP at 0.25 cm.
- ICP rejects a result that moves too far from PointNet or does not provide a meaningful fitness/RMSE improvement.
- An optional quarter-turn fallback tries eight $\pm90°$ horizontal orientation hypotheses (axis-aligned and diagonal) only after normal ICP is rejected and the crop has at least 200 observed points. It accepts a candidate only with clearly stronger geometric support; this targets upright-versus-sideways failures without slowing ordinary predictions.
- Symmetry-aware metrics are used for labelled data. Test-set output is qualitative because no ground-truth pose is provided.

# 6D Pose Estimation: PointNet + Guarded ICP

This project estimates an object's 6D camera-frame pose from RGB-D data, an object segmentation mask, and a canonical mesh. PointNet predicts a coarse pose; three-stage ICP refines it only when geometric checks show a meaningful improvement.

The latest controlled ablation uses the corrected validation cache, 1,705 valid object
instances across 236 scenes, and identical fixed point sampling for each learned model:

| Method | Mean rotation error | Mean translation error |
|---|---:|---:|
| PointNet | 3.37° | 0.23 cm |
| PointNet + guarded ICP | **2.58°** | **0.18 cm** |
| RGB--point fusion | 3.31° | 0.25 cm |
| RGB--point fusion + guarded ICP | 3.17° | 0.22 cm |

PointNet + guarded ICP also has the highest 5°/1 cm success rate (95.31%), versus
92.73% for PointNet and 91.79% for fusion + ICP. The RGB fusion initializer has a
slightly lower mean rotation error than PointNet but a lower success rate and does not
combine with ICP as effectively; PointNet + guarded ICP is therefore the main method.

These are the recorded validation results for the retained checkpoints; they do not
claim that every later augmentation or fine-tuning experiment improved the baseline.

For methodology, results, and the experiment plan, see [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).

## Project layout

The eight Python modules have distinct roles:

| File | Purpose |
|---|---|
| `preprocess.py` | Generate per-object point-cloud caches and the optional memory-mapped cache |
| `helpers.py` | Configuration, dataset loading, augmentation, caching, and evaluation helpers |
| `model.py` | PointNet and experimental models, plus pose and symmetry-aware losses |
| `train.py` | Training loop, validation, checkpoints, and TensorBoard metrics |
| `inference.py` | Labelled/test inference, error reports, and ablation comparisons |
| `icp.py` | Canonical-mesh registration and guarded ICP refinement |
| `visualization.py` | Scene overlays and comparison plots |
| `pose_pipeline.py` | Command-line entry point for the main workflows |

`see_data.ipynb` remains the data-exploration notebook. `pose_workflow.ipynb` is
the compact interactive training/evaluation workflow. The methodological details
and failure cases are in `TECHNICAL_REPORT.md`.

## Setup and commands

Install dependencies with `pip install -r requirements.txt`. Dataset paths and
default hyperparameters are set by `get_config()` in `helpers.py`; its dataset
paths currently point to this project's Windows data location and may need editing
on another machine. Run commands from the project root:

```bash
python pose_pipeline.py preprocess --split train
python pose_pipeline.py preprocess --split val
python pose_pipeline.py build-cache --split train
python pose_pipeline.py train --epochs 400 --batch_size 128
python pose_pipeline.py evaluate
python pose_pipeline.py ablations
python pose_pipeline.py visualize-failure --scene 2-6-3 --object mustard_bottle
python pose_pipeline.py visualize-icp --model pointnet
python pose_pipeline.py infer-test --max-scenes 5
```

Preprocessing may also be run for `--split test`. The memory-mapped cache is
optional; it reduces repeated Windows file-open/decompression overhead. Training
requires a CUDA-enabled PyTorch installation for practical speed. It reports
compact epoch-level loss, rotation, and translation metrics, saves the best
checkpoint to `model_weights/pointnet_new.pth` by default, and writes detailed
curves to TensorBoard. This default output is separate from the reported baseline.
Use `python pose_pipeline.py train --help` for model, checkpoint, fine-tuning,
and data-loader options.

To fine-tune the retained baseline without overwriting it, for example:

```bash
python pose_pipeline.py train --init_checkpoint model_weights/pointnet_occlusion_guarded_v1.pth --checkpoint_path model_weights/pointnet_finetuned_new.pth --lr 0.0001 --epochs 100
```

`evaluate` uses `model_weights/pointnet_occlusion_guarded_v1.pth` by default.
Pass `--checkpoint <path>` to evaluate another PointNet checkpoint; training a new
checkpoint does not automatically change the evaluation default. `ablations`
compares PointNet, PointNet + guarded ICP, RGB--point fusion, and RGB--point fusion
+ guarded ICP on the same validation instances, writing a summary and per-instance
CSV files under `outputs/`. The visualization commands render a selected failure
case or representative ICP improvements to `output_images/`. Test inference has no
ground truth; it produces qualitative predictions only.

The retained checkpoints are `pointnet_occlusion_guarded_v1.pth` (reported
baseline), `point_image_fusion_v1.pth` (RGB fusion ablation), and
`pointnet_occlusion_guarded_nightly_ft_v2.pth` (notebook fine-tuning selection).
The earlier nightly v1 file is still present but is not used by a current default
command. Correspondence, confidence, and multi-hypothesis model definitions remain
in `model.py` as negative/experimental ablations; their discarded checkpoints are
not needed for the main workflow.

## Evaluate in notebooks

Use `see_data.ipynb` to understand the RGB-D data, object masks, point-cloud
lifting, and coordinate frames. Use `pose_workflow.ipynb` for interactive training,
validation, hard-scene inspection, and GT/prediction overlays. Code-heavy logic
lives in the Python modules above rather than notebook cells.

## Core behavior

- PointNet samples 1,024 points per object, with nine features per point: XYZ, RGB, and normals.
- Rotation uses a 6D representation; translation is predicted relative to the observed-cloud centroid.
- ICP uses point-to-plane stages at 1.0 cm and 0.5 cm, followed by point-to-point ICP at 0.25 cm.
- ICP rejects a result that moves too far from PointNet or does not provide a meaningful fitness/RMSE improvement.
- An optional quarter-turn fallback tries eight $\pm90°$ horizontal orientation hypotheses (axis-aligned and diagonal) only after normal ICP is rejected and the crop has at least 200 observed points. It accepts a candidate only with clearly stronger geometric support; this targets upright-versus-sideways failures without slowing ordinary predictions.
- Symmetry-aware metrics are used for labelled data. Test-set output is qualitative because no ground-truth pose is provided.

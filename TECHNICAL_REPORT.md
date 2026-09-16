# Technical Report: 6D Pose Estimation with PointNet and Guarded ICP

## 1. Goal

Given an RGB-D image, an object segmentation mask, and a canonical mesh for a known object class, estimate the object-to-camera pose

$$T_{co} = \begin{bmatrix}R & t\\0 & 1\end{bmatrix}, \qquad R \in SO(3),\; t \in \mathbb{R}^3.$$

The system uses PointNet to predict a coarse pose from the visible object point cloud, then uses local geometric registration to refine that pose when the refinement is reliable.

## 2. Data and Coordinate Frames

- **Camera frame:** Depth pixels are back-projected with the camera intrinsics $K$.
- **Object frame:** Canonical mesh coordinates, scaled using the per-instance metadata scale.
- **Registration:** The scaled object mesh is aligned directly to the observed object point cloud in the camera frame.
- **Ground truth:** For labelled train/validation scenes, $T_{co}=T_{wc}T_{ow}$.

Each input cloud is extracted using the object label mask, voxel-downsampled at 5 mm, supplied with RGB and camera-oriented normals, centred, and sampled to 1,024 points. The network input is therefore $N\times9$: XYZ, RGB, and normal components.

The train, validation, and test point-cloud caches have been generated from their matching data roots. This avoids the earlier issue where a cache from one dataset was paired with images from another dataset, which produced misleading overlays and invalid evaluation results.

## 3. Model

The current learned baseline is PointNet with class conditioning:

1. Shared pointwise MLP: $9\rightarrow64\rightarrow128\rightarrow1024$.
2. Max pooling produces a permutation-invariant global feature.
3. A learned object-ID embedding is concatenated with that feature.
4. Separate heads predict a 6D rotation representation and a centroid-relative translation residual.

The 6D rotation prediction is converted to an orthonormal rotation matrix. Translation is reconstructed as

$$t=\bar{p}+t_{res},$$

where $\bar{p}$ is the observed-cloud centroid.

Training combines point-matching, symmetry-aware geodesic rotation, and translation losses. Symmetry is taken from the project symmetry table rather than the CSV geometry label, so equivalent poses are not penalized as errors.

### Occlusion and pose-diversity augmentation

The current training code now includes partial-view augmentation. With probability 0.65, it removes one contiguous side of the centred cloud and keeps 55–90% of its points before resampling. The retained points are re-centred and the translation residual is adjusted to the new centroid, preserving the coordinate convention used at inference. This represents the blocked object surfaces common in difficult scenes. Standard small XYZ jitter and rigid pose perturbations remain enabled during training.

The initial small rigid perturbation is limited to $\pm30°$ on each camera axis. After inspecting scene `2-6-3`, we found that a laid-down mustard bottle and bleach cleanser were instead predicted upright (about $98°$ and $86°$ rotation error, respectively), despite 1,078 and 1,537 valid observed points. The failure is therefore not caused by sparse crops; it is a large orientation-distribution gap. Guarded ICP correctly left the poses unchanged because its correction radius is local.

To address this, training now applies an additional $\pm90°$ quarter-turn around the camera $x$ or $y$ axis with probability 0.50, plus a $180°$ horizontal-axis flip with probability 0.15. It is applied identically to centred XYZ, normals, the rotation target, and the centroid-relative translation residual, generating physically relevant side-lying and flipped orientations while preserving the established pose convention. This synthetic orientation enrichment is preferable to trying to label every mesh's local "up" direction, which is not consistent across object coordinate systems.

These augmentation changes require retraining and evaluation before their effect can be reported as an improvement.

## 4. Guarded ICP Refinement

PointNet's prediction initializes three-stage ICP:

| Stage | Objective | Distance threshold |
|---|---|---|
| 1 | Point-to-plane | 1.0 cm |
| 2 | Point-to-plane | 0.5 cm |
| 3 | Point-to-point | 0.25 cm |

Target normals are oriented toward the camera. Point-to-plane ICP obtains a coarse surface alignment; the final point-to-point stage reduces sliding along broad planar faces.

ICP is **guarded**. Before refinement, the system evaluates the PointNet pose at the same registration scale. A refined pose is accepted only if it:

- changes rotation by at most the configured limit (10° by default);
- changes translation by at most 1 cm;
- does not materially reduce fitness or worsen RMSE; and
- provides a meaningful improvement in fitness or RMSE.

Otherwise the PointNet pose is retained. This is intended to prevent an apparently plausible ICP local minimum from degrading correct predictions for heavily occluded or symmetric objects.

### Quarter-turn fallback for large orientation ambiguity

Normal guarded ICP is deliberately local and therefore cannot correct a PointNet pose that is roughly 90° wrong. An optional fallback is triggered only when normal ICP is rejected and the observed crop has at least 200 points. It tests eight PointNet-derived hypotheses ($\pm90°$ about axis-aligned and diagonal horizontal camera axes), locally refines each, and accepts one only if its registration fitness clearly exceeds the original pose while RMSE remains competitive. This is a targeted mechanism for laid-down objects; severely occluded crops remain unmodified because they lack enough evidence for trustworthy hypothesis selection.

Pure ICP (FPFH/RANSAC followed by local ICP) is retained only as a classical baseline. It is not used in the main learned pipeline because global correspondences can be too sparse for small, symmetric, or partially visible objects.

## 5. Evaluation Protocol

Only labelled train/validation data is used for pose-error metrics. Objects with fewer than 50 valid points are excluded.

- **Rotation error:** symmetry-aware angular error in degrees.
- **Translation error:** Euclidean camera-frame error in centimetres.
- **Success criterion:** rotation $\leq5°$ and translation $\leq1$ cm.

The notebook evaluates PointNet-only and PointNet+ICP on exactly the same instances, exports one row per scene-object pair, and visualizes the hardest scenes side by side. In ICP-improvement examples, ground truth is dark green, PointNet is orange-red, and ICP is bright green, intentionally making the GT/ICP relationship easy to compare. The plots include PointNet-to-ICP rotation and translation errors. The 3D overlay is interactive: the observed cloud is rendered once with all three bounding-box wireframes, and can be freely rotated and zoomed in the notebook.

## 6. Latest Validated Baseline Results

The most recent full validation run used the corrected validation cache and evaluated 1,705 valid object instances across 236 scenes.

| Method | Mean rotation error | Mean translation error |
|---|---:|---:|
| PointNet | 3.57° | 0.21 cm |
| PointNet + three-stage ICP | **2.73°** | **0.19 cm** |

ICP reduced mean rotation error by 0.84° (about 24%) and translation error by 0.02 cm (about 10%). Translation is already strong; residual failures are expected to be dominated by ambiguous rotation, limited visible geometry, and occlusion.

These numbers are the established baseline checkpoint results. They were recorded **before retraining with the new partial-view augmentation and before a new full evaluation of the guarded-ICP code**. Future report updates must list those experiments separately rather than treating them as measured improvements.

## 7. Planned Ablation Study

All ablations will use the same corrected validation split and report mean/median rotation and translation errors, 5°/1 cm success rate, per-object results, and per-scene results.

| Experiment | Status | Purpose |
|---|---|---|
| PointNet | Completed baseline | Learned coarse pose only |
| PointNet + ICP | Completed baseline | Measure geometric local refinement |
| PointNet + guarded ICP | Pending evaluation | Measure whether confidence gating prevents ICP regressions |
| PointNet + quarter-turn ICP fallback | Pending evaluation | Recover geometrically supported upright-versus-sideways failures without retraining |
| PointNet with partial-view + 90°/180° orientation augmentation + guarded ICP | Pending retraining/evaluation | Target occlusion, side-lying, and flipped-orientation failures |
| PointNet++ | Future ablation | Test whether local hierarchical features improve partial-object pose estimation |

## 8. Implementation Notes

- PointNet inference can run on GPU when available; Open3D ICP runs on CPU.
- ICP samples are processed with a thread pool and canonical object point clouds are cached to avoid repeated mesh loading.
- The test set has no ground-truth poses. It is used for qualitative box overlays and ICP proxy diagnostics only, not rotation/translation accuracy claims.
- Pipeline inference lives in `test_inference.py`; per-instance diagnostics live in `error_analysis.py`; reusable labelled-evaluation workflows live in `evaluation_workflows.py`; and all 2D/interactive-3D rendering lives in `pose_visualization.py`. The notebook section beginning at **ICP** is intentionally limited to concise orchestration cells.
- The notebook has one shared `CHECKPOINT_PATH` variable and a guarded **Train a new checkpoint** cell. Validation, level-2, and test inference all use that same selected checkpoint.
- Training exposes live batch-level loss bars and an epoch-level summary bar; notebook training runs in the active kernel so progress renders directly in the notebook output. It uses the current `torch.amp.autocast` API when CUDA is active.
- GPU transfer uses pinned-memory non-blocking copies, cuDNN benchmarking, and high-precision matrix-multiplication settings. The notebook's next-run defaults use a 256-sample batch and 12 data-loader workers to better utilize the RTX 4090.
- The notebook includes a one-time CUDA PyTorch setup cell. Training explicitly stops if CUDA is requested but unavailable, avoiding an accidental CPU-only run.
- Training writes TensorBoard summaries under `runs/`; the notebook dashboard resolves that directory absolutely and refreshes live loss and pose-error curves every five seconds.
- Cache generation exposes a named live progress bar with throughput and point-count diagnostics, and safely resumes by skipping existing completed sample files.
- The original training cache stores one compressed NPZ per object, causing repeated small-file and decompression overhead under Windows workers. An optional packed memory-mapped cache now stores contiguous file-backed arrays plus point offsets. Workers map the same OS-managed pages while preserving per-epoch random sampling and augmentation, reducing I/O without duplicating the entire cache per worker.
- The notebook includes a nightly fine-tuning mode. It initializes a separate checkpoint from the completed best model, resets optimizer/scheduler state, and performs a new 400-epoch cosine schedule at a lower $10^{-4}$ learning rate. The initial checkpoint is retained as the first candidate, so fine-tuning cannot leave the chosen output path without a usable model.
- Fine-tuning additionally saves a checkpoint every 25 epochs. This is necessary for targeted experiments: a model that improves rare laid-down or heavily occluded objects can slightly worsen the global validation average, so it would otherwise be discarded by best-global-loss-only selection. Snapshot models are evaluated separately on the predefined hard-scene subset.

## 9. Next Step

Retrain PointNet with partial-view augmentation, then rerun the full validation and error-report notebook cells. The new results should be compared directly with the baseline table above, especially on the worst scenes and object classes.

# Ablation Studies Log

Use this table to record the results of your experiments.

| Experiment ID | Method | Configuration / Hyperparameters | Pass Rate (%) | Avg Rot Err (deg) | Avg Trans Err (cm) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | PointNet Only | `eval_pn_icp.py --no_icp` | 72.26% | 4.70 | 0.47 | Pure PointNet prediction without refinement. |
| Exp 1 | PointNet + ICP | `eval_pn_icp.py` (Default 3-stage) | 78.36% | 6.42 | 0.47 | Full pipeline with robust 3-stage ICP. Pass rate improved, but avg rot error increased (likely due to outliers/flips). |
| **Exp 2** | ICP Stages | 1-Stage ICP (Point-to-Plane only) | 50.85% | 8.96 | 0.64 | Single coarse stage degrades performance significantly. PointNet init is better than coarse ICP convergence. |
| **Exp 3** | ICP Threshold | `max_corr_dist` = 0.05 (5cm) | 67.86% | 11.18 | 0.57 | Relaxed threshold hurt performance. It likely allowed ICP to latch onto incorrect geometry (outliers/clutter), pulling good PointNet predictions into bad local minima. |
| **Exp 4** | ICP Threshold | `max_corr_dist` = 0.01 (1cm) | 76.36% | 5.52 | 0.46 | Tighter threshold reduces rotation error (fixes "flipping" issues like the can), but slightly lowers pass rate compared to 2cm. It rejects bad matches but also fails to fix PointNet predictions that are >1cm off. |
| **Exp 5** | Safety Check | `max_rot_change` = 30 deg | 74.90% | 5.80 | 0.44 | Attempted to combine 2cm threshold with a safety check. Pass rate dropped because the check naively rejects valid symmetric flips (e.g., 180° corrections for boxes) that ICP correctly found. |
| **Baseline 2** | ICP Only | `icp.py` (Centroid Init) | 16.58% | 70.85 | 1.61 | Pure geometric approach fails without good rotation initialization. Confirms PointNet is critical. |
| **Exp 6** | Input Points | `NUM_POINTS` = 2048 | | | | Train/Eval with denser point clouds. |
| **Exp 5** | Features | XYZ Only (No RGB/Normals) | | | | Retrain model with `in_channels=0` (or 3). |
| **Exp 6** | Symmetry | No Symmetry Loss | | | | Retrain without `symmetry_aware_geodesic_loss`. |
| **Exp 7** | Occlusion | No Occlusion Filtering | | | | Disable the label-based occlusion check in `eval.py`. |

## Detailed Notes

### Baseline: PointNet Only
- **Command:** `python eval_pn_icp.py --no_icp`
- **Goal:** Establish the lower bound performance of the neural network alone.

### Exp 1: PointNet + ICP (3-Stage)
- **Command:** `python eval_pn_icp.py`
- **Goal:** Measure the improvement provided by geometric refinement.

### Exp 2: ICP Stages (Ablation)
- **Modification:** Edit `refine_icp` in `eval_pn_icp.py` to comment out Stage 2 and 3.
- **Goal:** Determine if the multi-stage approach is necessary or if a simple ICP suffices.

### Exp 4: Point Density
- **Modification:** Change `NUM_POINTS` in `train.py` and `eval_pn_icp.py`.
- **Goal:** See if more points provide better shape context for the PointNet encoder.

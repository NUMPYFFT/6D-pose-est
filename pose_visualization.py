"""Reusable 2D and interactive-3D pose visualizations."""

from collections import defaultdict
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

import utils


# Color convention used throughout the evaluation notebook.
GT_COLOR = (0, 100, 0)          # dark green
POINTNET_COLOR = (235, 80, 60)  # orange-red
ICP_COLOR = (0, 255, 0)         # bright green


def scene_name(sample):
    """Return the dataset scene identifier for one inference sample."""
    return os.path.basename(sample["rgb_path"]).split("_")[0]


def _load_rgb(rgb_path):
    image = cv2.imread(rgb_path)
    if image is None:
        raise FileNotFoundError(f"Could not load RGB image: {rgb_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _draw_box(image, sample, rotation_key, translation_key, color, thickness=2):
    size = sample["obj_dims"] * sample["scale"]
    utils.draw_projected_box3d(
        image, sample[translation_key], size, sample[rotation_key],
        np.eye(4), sample["K"], color=color, thickness=thickness,
    )


def _samples_by_scene(samples):
    grouped = defaultdict(list)
    for sample in samples:
        grouped[scene_name(sample)].append(sample)
    return grouped


def visualize_test_predictions(samples):
    """Draw final test-set predictions. Test data has no GT pose overlays."""
    for current_scene, scene_samples in _samples_by_scene(samples).items():
        image = _load_rgb(scene_samples[0]["rgb_path"])
        for sample in scene_samples:
            _draw_box(image, sample, "pred_R", "pred_t", ICP_COLOR)
        plt.figure(figsize=(14, 8), dpi=160)
        plt.imshow(image)
        plt.title(f"{current_scene} — test prediction: PointNet + guarded ICP")
        plt.axis("off")
        plt.show()


def visualize_worst_validation_scenes(samples, by_scene, max_scenes=5):
    """Show GT, PointNet, and ICP overlays for the highest-error scenes."""
    grouped = _samples_by_scene(samples)
    selected_scenes = [name for name in by_scene.head(max_scenes)["scene"] if name in grouped]

    for current_scene in selected_scenes:
        scene_samples = grouped[current_scene]
        image = _load_rgb(scene_samples[0]["rgb_path"])
        gt_image, pointnet_image, icp_image = image.copy(), image.copy(), image.copy()
        for sample in scene_samples:
            _draw_box(gt_image, sample, "gt_R", "gt_t", GT_COLOR)
            _draw_box(pointnet_image, sample, "pointnet_R", "pointnet_t", POINTNET_COLOR)
            _draw_box(icp_image, sample, "pred_R", "pred_t", ICP_COLOR)

        fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=180)
        for axis, frame, title in zip(
            axes, (gt_image, pointnet_image, icp_image),
            ("Ground truth (dark green)", "PointNet (orange-red)", "PointNet + ICP (bright green)"),
        ):
            axis.imshow(frame)
            axis.set_title(title)
            axis.axis("off")
        fig.suptitle(f"Hard validation scene: {current_scene}")
        fig.tight_layout()
        plt.show()


def select_icp_improvement_examples(report, max_examples=5,
                                    min_rotation_improvement_deg=1.0,
                                    min_translation_improvement_cm=0.10):
    """Select distinct scenes where ICP materially improves the initial pose."""
    examples = report.copy()
    examples["improvement_score"] = (
        examples["icp_rot_improvement_deg"].clip(lower=0)
        + 5.0 * examples["icp_trans_improvement_cm"].clip(lower=0)
    )
    examples = examples[
        (examples["icp_rot_improvement_deg"] >= min_rotation_improvement_deg)
        | (examples["icp_trans_improvement_cm"] >= min_translation_improvement_cm)
    ].sort_values("improvement_score", ascending=False)
    return examples.drop_duplicates("scene").head(max_examples).reset_index(drop=True)


def _zoom_bounds(label_path, obj_id, image_shape):
    label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
    if label is None:
        return None
    ys, xs = np.where(label == obj_id)
    if not len(xs):
        return None
    margin = max(20, int(0.35 * max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)))
    return (
        max(0, xs.min() - margin), min(image_shape[1], xs.max() + margin + 1),
        max(0, ys.min() - margin), min(image_shape[0], ys.max() + margin + 1),
    )


def _box_corners(rotation, translation, size):
    half = np.asarray(size) / 2.0
    local = np.array([
        [-half[0], -half[1], -half[2]], [half[0], -half[1], -half[2]],
        [half[0], half[1], -half[2]], [-half[0], half[1], -half[2]],
        [-half[0], -half[1], half[2]], [half[0], -half[1], half[2]],
        [half[0], half[1], half[2]], [-half[0], half[1], half[2]],
    ])
    return local @ rotation.T + translation


def _show_interactive_box_overlay(target, example):
    """Render one rotatable point-cloud view with all three pose boxes."""
    import plotly.graph_objects as go

    observed = target["obs_pts"]
    poses = (
        (target["gt_R"], target["gt_t"], "darkgreen", "Ground truth"),
        (target["pointnet_R"], target["pointnet_t"], "orangered", "PointNet"),
        (target["pred_R"], target["pred_t"], "limegreen", "ICP"),
    )
    boxes = [_box_corners(rotation, translation, target["obj_dims"] * target["scale"])
             for rotation, translation, _, _ in poses]
    all_points = np.vstack([observed, *boxes])
    lower, upper = all_points.min(axis=0), all_points.max(axis=0)
    centre = (lower + upper) / 2.0
    half_range = max((upper - lower).max() / 2.0, 0.02) * 1.10

    figure = go.Figure()
    figure.add_trace(go.Scatter3d(
        x=observed[:, 0], y=observed[:, 1], z=observed[:, 2],
        mode="markers", name="Observed crop",
        marker=dict(size=2.5, color="lightgray", opacity=0.5),
    ))
    edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
             (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
    for (_, _, color, label), corners in zip(poses, boxes):
        for edge_index, (start, end) in enumerate(edges):
            figure.add_trace(go.Scatter3d(
                x=corners[[start, end], 0], y=corners[[start, end], 1], z=corners[[start, end], 2],
                mode="lines", name=label, legendgroup=label, showlegend=edge_index == 0,
                line=dict(color=color, width=7),
            ))

    figure.update_layout(
        title=(
            f"Interactive 3D box overlay: {example['scene']} — {example['object_name']}<br>"
            f"PointNet: {example['pointnet_rot_deg']:.1f}° / {example['pointnet_trans_cm']:.2f} cm "
            f"→ ICP: {example['final_rot_deg']:.1f}° / {example['final_trans_cm']:.2f} cm"
        ),
        height=760, margin=dict(l=0, r=0, t=80, b=0), legend=dict(itemsizing="constant"),
        scene=dict(
            xaxis=dict(range=[centre[0] - half_range, centre[0] + half_range], visible=False),
            yaxis=dict(range=[centre[1] - half_range, centre[1] + half_range], visible=False),
            zaxis=dict(range=[centre[2] - half_range, centre[2] + half_range], visible=False),
            aspectmode="cube",
        ),
    )
    figure.show()


def visualize_icp_improvement_examples(samples, report, max_examples=5, show_3d=True):
    """Plot high-resolution 2D and optional interactive 3D ICP improvements.

    Returns the selected diagnostic rows so the notebook can display or export them.
    """
    examples = select_icp_improvement_examples(report, max_examples=max_examples)
    if examples.empty:
        raise RuntimeError("No notable ICP improvements found. Rerun validation or lower the thresholds.")

    grouped = _samples_by_scene(samples)
    for _, example in examples.iterrows():
        scene_samples = grouped[example["scene"]]
        target = next(sample for sample in scene_samples if sample["obj_id"] == example["object_id"])
        image = _load_rgb(target["rgb_path"])
        gt_image, pointnet_image, icp_image = image.copy(), image.copy(), image.copy()

        for sample in scene_samples:
            thickness = 3 if sample["obj_id"] == target["obj_id"] else 1
            _draw_box(gt_image, sample, "gt_R", "gt_t", GT_COLOR, thickness)
            _draw_box(pointnet_image, sample, "pointnet_R", "pointnet_t", POINTNET_COLOR, thickness)
            _draw_box(icp_image, sample, "pred_R", "pred_t", ICP_COLOR, thickness)

        title = (
            f"{example['scene']} — {example['object_name']}: rotation "
            f"{example['pointnet_rot_deg']:.1f}° → {example['final_rot_deg']:.1f}°, "
            f"translation {example['pointnet_trans_cm']:.2f} → {example['final_trans_cm']:.2f} cm"
        )
        fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=180)
        for axis, frame, panel_title in zip(
            axes, (gt_image, pointnet_image, icp_image),
            ("Ground truth (dark green)", "PointNet (orange-red)", "ICP (bright green)"),
        ):
            axis.imshow(frame)
            axis.set_title(panel_title)
            axis.axis("off")
        fig.suptitle(title)
        fig.tight_layout()
        plt.show()

        label_path = target["rgb_path"].replace("_color_kinect.png", "_label_kinect.png")
        bounds = _zoom_bounds(label_path, target["obj_id"], image.shape)
        if bounds is not None:
            x0, x1, y0, y1 = bounds
            fig, axes = plt.subplots(1, 3, figsize=(21, 7), dpi=220)
            for axis, frame, panel_title in zip(
                axes, (gt_image, pointnet_image, icp_image),
                ("GT zoom (dark green)", "PointNet zoom (orange-red)", "ICP zoom (bright green)"),
            ):
                axis.imshow(frame[y0:y1, x0:x1], interpolation="nearest")
                axis.set_title(panel_title)
                axis.axis("off")
            fig.suptitle(title)
            fig.tight_layout()
            plt.show()

        if show_3d:
            _show_interactive_box_overlay(target, example)

    return examples

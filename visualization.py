"""Reusable 2D and interactive-3D pose visualizations."""

from collections import defaultdict
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

import helpers as utils


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


"""Create a 2x2 same-scene pose visualization for the main ablations."""

import argparse
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch


import inference


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_POINTNET = PROJECT_ROOT / "model_weights" / "pointnet_occlusion_guarded_v1.pth"
DEFAULT_FUSION = PROJECT_ROOT / "model_weights" / "point_image_fusion_v1.pth"
FUSION_COLOR = (145, 85, 210)  # purple distinguishes the RGB--point initializer


def _set_seed_main_ablation(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_scene(scene, model_name, checkpoint, seed):
    _set_seed_main_ablation(seed)
    samples, _ = inference.run_labelled_inference(
        split="val", scene_names={scene}, model_name=model_name,
        checkpoint_path=str(checkpoint), refine_with_icp=True,
    )
    return samples


def find_target(samples, object_name):
    for sample in samples:
        if sample["obj_name"] == object_name:
            return sample
    available = sorted({sample["obj_name"] for sample in samples})
    raise ValueError(f"Object {object_name!r} was not found. Available: {available}")


def panel_image(target, rotation_key, translation_key, prediction_color):
    """Render just the target's GT and predicted boxes on its original RGB image."""
    image = _load_rgb(target["rgb_path"])
    _draw_box(image, target, "gt_R", "gt_t", GT_COLOR, thickness=3)
    _draw_box(image, target, rotation_key, translation_key, prediction_color, thickness=3)
    label_path = target["rgb_path"].replace("_color_kinect.png", "_label_kinect.png")
    bounds = _zoom_bounds(label_path, target["obj_id"], image.shape)
    if bounds is None:
        return image
    x0, x1, y0, y1 = bounds
    return image[y0:y1, x0:x1]


def pose_errors(target, rotation_key, translation_key):
    import model as loss_utils
    rot = loss_utils.compute_symmetry_aware_loss(
        target[rotation_key], target["gt_R"], target["sym"]
    )
    trans = np.linalg.norm(target[translation_key] - target["gt_t"]) * 100.0
    return rot, trans


def visualize_main_ablations():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="2-6-3")
    parser.add_argument("--object", dest="object_name", default="mustard_bottle")
    parser.add_argument("--pointnet_checkpoint", default=str(DEFAULT_POINTNET))
    parser.add_argument("--fusion_checkpoint", default=str(DEFAULT_FUSION))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output_dir", default="output_images")
    parser.add_argument("--show", action="store_true", help="Display the figure after saving it.")
    args = parser.parse_args()
    for checkpoint in (args.pointnet_checkpoint, args.fusion_checkpoint):
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    pointnet = find_target(
        infer_scene(args.scene, "pointnet", args.pointnet_checkpoint, args.seed),
        args.object_name,
    )
    fusion = find_target(
        infer_scene(args.scene, "fusion", args.fusion_checkpoint, args.seed),
        args.object_name,
    )
    panels = (
        ("PointNet", pointnet, "pointnet_R", "pointnet_t", POINTNET_COLOR),
        ("PointNet + guarded ICP", pointnet, "pred_R", "pred_t", ICP_COLOR),
        ("RGB-point fusion", fusion, "pointnet_R", "pointnet_t", FUSION_COLOR),
        ("RGB-point fusion + guarded ICP", fusion, "pred_R", "pred_t", ICP_COLOR),
    )

    figure, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=220, constrained_layout=True)
    for axis, (label, target, rotation_key, translation_key, color) in zip(axes.flat, panels):
        rotation_error, translation_error = pose_errors(target, rotation_key, translation_key)
        axis.imshow(panel_image(target, rotation_key, translation_key, color))
        axis.set_title(f"{label}\n{rotation_error:.1f}° rotation | {translation_error:.2f} cm translation")
        axis.axis("off")
    figure.suptitle(
        f"Same-scene ablation: {args.scene} — {args.object_name}\n"
        "GT: dark green | initializer: orange/red or purple | ICP: bright green",
        fontsize=15,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"ablation_visual_{args.scene}_{args.object_name}.png"
    figure.savefig(output, dpi=220, bbox_inches="tight")
    print(f"Saved side-by-side visualization: {output}")
    if args.show:
        plt.show()
    plt.close(figure)




"""Render representative, measurable ICP gains for PointNet or RGB--point fusion."""

import argparse
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import model as loss_utils

import inference


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_SPECS = {
    "pointnet": {
        "checkpoint": PROJECT_ROOT / "model_weights" / "pointnet_occlusion_guarded_v1.pth",
        "report": PROJECT_ROOT / "outputs" / "ablation_pointnet_instances.csv",
        "color": POINTNET_COLOR,
        "title": "PointNet",
    },
    "fusion": {
        "checkpoint": PROJECT_ROOT / "model_weights" / "point_image_fusion_v1.pth",
        "report": PROJECT_ROOT / "outputs" / "ablation_rgb_point_fusion_instances.csv",
        "color": (145, 85, 210),
        "title": "RGB-point fusion",
    },
}


def _set_seed_icp_improvements(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_candidates(report, count):
    report = report.copy()
    report["score"] = (
        report["icp_rot_improvement_deg"].clip(lower=0)
        + 5.0 * report["icp_trans_improvement_cm"].clip(lower=0)
    )
    eligible = report[
        (report["icp_rot_improvement_deg"] >= 2.0)
        & (report["final_rot_deg"] <= 5.0)
        & (report["final_trans_cm"] <= 1.0)
    ].sort_values("score", ascending=False)
    candidates = eligible.drop_duplicates("scene").head(max(count * 8, 12))
    if len(candidates) < count:
        raise RuntimeError("Not enough clear ICP improvements in the ablation report.")
    return candidates


def target_from_scene(scene, object_name, model_name, checkpoint, seed):
    _set_seed_icp_improvements(seed)
    samples, _ = inference.run_labelled_inference(
        split="val", scene_names={scene}, model_name=model_name,
        checkpoint_path=str(checkpoint), refine_with_icp=True,
    )
    for sample in samples:
        if sample["obj_name"] == object_name:
            return sample
    raise ValueError(f"{object_name!r} was not found in scene {scene!r}.")


def draw_panel(target, rotation_key, translation_key, color):
    image = _load_rgb(target["rgb_path"])
    _draw_box(image, target, "gt_R", "gt_t", GT_COLOR, thickness=3)
    _draw_box(image, target, rotation_key, translation_key, color, thickness=3)
    label_path = target["rgb_path"].replace("_color_kinect.png", "_label_kinect.png")
    bounds = _zoom_bounds(label_path, target["obj_id"], image.shape)
    if bounds is not None:
        x0, x1, y0, y1 = bounds
        image = image[y0:y1, x0:x1]
    rotation = loss_utils.compute_symmetry_aware_loss(
        target[rotation_key], target["gt_R"], target["sym"]
    )
    translation = np.linalg.norm(target[translation_key] - target["gt_t"]) * 100.0
    return image, rotation, translation


def visualize_icp_improvements():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_SPECS, required=True)
    parser.add_argument("--examples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    # ``outputs/`` contains evaluation CSVs but may be read-only under some
    # Windows workspace ACLs. Rendered figures belong in the writable image folder.
    parser.add_argument("--output_dir", default="output_images")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    spec = MODEL_SPECS[args.model]
    if not spec["report"].is_file():
        raise FileNotFoundError(
            f"Missing {spec['report'].name}. Run compare_main_ablations.py first."
        )
    report = pd.read_csv(spec["report"])
    candidates = select_candidates(report, args.examples)
    # Dataset point sampling is stochastic. Recheck candidates using the exact
    # sampled cloud and pose that will be drawn, so a visual never claims an
    # improvement that is absent from its own displayed prediction.
    examples, targets = [], []
    for candidate_index, (_, candidate) in enumerate(candidates.iterrows()):
        target = target_from_scene(
            candidate["scene"], candidate["object_name"], args.model,
            spec["checkpoint"], args.seed + candidate_index,
        )
        init_rot = loss_utils.compute_symmetry_aware_loss(
            target["pointnet_R"], target["gt_R"], target["sym"])
        final_rot = loss_utils.compute_symmetry_aware_loss(
            target["pred_R"], target["gt_R"], target["sym"])
        init_trans = np.linalg.norm(target["pointnet_t"] - target["gt_t"]) * 100.0
        final_trans = np.linalg.norm(target["pred_t"] - target["gt_t"]) * 100.0
        if (init_rot - final_rot >= 2.0 and final_rot <= 5.0 and final_trans <= 1.0):
            candidate = candidate.copy()
            candidate["display_init_rot_deg"] = init_rot
            candidate["display_final_rot_deg"] = final_rot
            candidate["display_init_trans_cm"] = init_trans
            candidate["display_final_trans_cm"] = final_trans
            examples.append(candidate)
            targets.append(target)
        if len(examples) == args.examples:
            break
    if len(examples) < args.examples:
        raise RuntimeError(
            "Could not reproduce enough clear ICP improvements with the current point samples. "
            "Try --examples 1."
        )

    figure, axes = plt.subplots(args.examples, 3, figsize=(15, 5 * args.examples), dpi=220)
    axes = np.atleast_2d(axes)
    for row_index, (example, target) in enumerate(zip(examples, targets)):
        images = [
            (_load_rgb(target["rgb_path"]), "Ground truth", "gt_R", "gt_t", GT_COLOR),
            (None, f"{spec['title']} initializer", "pointnet_R", "pointnet_t", spec["color"]),
            (None, "Guarded ICP", "pred_R", "pred_t", ICP_COLOR),
        ]
        for col_index, (base_image, label, r_key, t_key, color) in enumerate(images):
            if label == "Ground truth":
                image = base_image.copy()
                _draw_box(image, target, r_key, t_key, color, thickness=3)
                label_path = target["rgb_path"].replace("_color_kinect.png", "_label_kinect.png")
                bounds = _zoom_bounds(label_path, target["obj_id"], image.shape)
                if bounds is not None:
                    x0, x1, y0, y1 = bounds
                    image = image[y0:y1, x0:x1]
                title = label
            else:
                image, rot, trans = draw_panel(target, r_key, t_key, color)
                title = f"{label}\n{rot:.1f}° | {trans:.2f} cm"
            axes[row_index, col_index].imshow(image)
            axes[row_index, col_index].set_title(title)
            axes[row_index, col_index].axis("off")
        axes[row_index, 0].set_ylabel(
            f"{example['scene']}\n{example['object_name']}\n"
            f"ICP gain shown: {example['display_init_rot_deg'] - example['display_final_rot_deg']:.1f}°",
            fontsize=11,
        )
    figure.suptitle(
        f"Representative guarded-ICP improvements: {spec['title']}\n"
        "GT: dark green | initializer: model color | ICP: bright green",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"icp_improvements_{args.model}.png"
    figure.savefig(output, dpi=220, bbox_inches="tight")
    print("Selected examples:")
    display_table = pd.DataFrame(examples)[[
        "scene", "object_name", "display_init_rot_deg", "display_final_rot_deg",
        "display_init_trans_cm", "display_final_trans_cm",
    ]]
    print(display_table.to_string(index=False))
    print(f"Saved: {output}")
    if args.show:
        plt.show()
    plt.close(figure)

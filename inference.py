"""Learned pose initialization, guarded ICP, evaluation, and test inference."""

from pathlib import Path
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
# Use the stable standard renderer instead of the VS Code notebook widget.
from tqdm import tqdm

import helpers as config
import icp
import pose_losses
import helpers as utils
from helpers import PoseDataset
from model import build_pose_model

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TEST_DATA_DIR = Path(r"C:\Users\Owner\Desktop\Fall2023\CSE275\6D Pose\testing_data_pose_filtered\testing_data\v2.2")
DEFAULT_TEST_SPLIT_DIR = PROJECT_ROOT / "training_data_filtered" / "training_data" / "splits" / "v2"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "model_weights" / "pointnet_occlusion_guarded_v1.pth"


def _predict_pose(model, batch, points, object_ids, device, model_name):
    if model_name == "fusion":
        rotation, translation = model(
            points, object_ids,
            batch["image"].to(device, non_blocking=True),
            batch["pixel_coords"].to(device, non_blocking=True),
        )
        return rotation, translation, None
    rotation, translation = model(points, object_ids)
    return rotation, translation, None


def _scene_subset(dataset, max_scenes, scene_prefix=None):
    """Return dataset indices for complete scenes, optionally filtered by prefix."""
    selected_scenes, selected_indices = [], []
    for dataset_idx, (scene_idx, _) in enumerate(dataset.samples):
        scene_name = os.path.basename(dataset.rgb_files[scene_idx]).split("_")[0]
        if scene_prefix is not None and not scene_name.startswith(scene_prefix):
            continue
        if scene_name not in selected_scenes:
            if max_scenes is not None and len(selected_scenes) >= max_scenes:
                continue
            selected_scenes.append(scene_name)
        selected_indices.append(dataset_idx)
    return selected_indices, selected_scenes


def run_test_inference(max_scenes=5, scene_prefix=None, test_data_dir=DEFAULT_TEST_DATA_DIR,
                       test_split_dir=DEFAULT_TEST_SPLIT_DIR,
                       checkpoint_path=DEFAULT_CHECKPOINT, refine_with_icp=True,
                       quarter_turn_fallback=False, model_name="pointnet"):
    """Predict test poses with the selected model and optional ICP, without GT metrics."""
    args = config.get_config(args=[])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    objects_df = pd.read_csv(PROJECT_ROOT / args.objects_csv)
    dataset = PoseDataset(
        "test", str(test_data_dir), str(test_split_dir), num_points=args.num_points,
        return_image=model_name == "fusion", image_size=args.image_size,
    )
    indices, scene_names = _scene_subset(dataset, max_scenes, scene_prefix=scene_prefix)
    if not indices:
        prefix_text = repr(scene_prefix) if scene_prefix is not None else "the requested selection"
        raise ValueError(f"No test objects found for {prefix_text}.")
    loader = DataLoader(Subset(dataset, indices), batch_size=min(args.batch_size, len(indices)),
                        shuffle=False, num_workers=0)

    model = build_pose_model(model_name, num_classes=args.num_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    samples = []
    with torch.no_grad():
        initializer = "RGB fusion" if model_name == "fusion" else "PointNet"
        for batch in tqdm(loader, desc=f"{initializer} test inference", unit="batch"):
            points = batch["points"].to(device)
            object_ids = batch["obj_id"].to(device)
            rot6d, pred_t_residual, auxiliary = _predict_pose(
                model, batch, points, object_ids, device, model_name
            )
            pred_Rs = utils.rotation_6d_to_matrix(rot6d).cpu().numpy()
            pred_t_residuals = pred_t_residual.cpu().numpy()
            for i, rgb_path in enumerate(batch["rgb_path"]):
                obj_id = batch["obj_id"][i].item()
                with open(rgb_path.replace("_color_kinect.png", "_meta.pkl"), "rb") as handle:
                    meta = pickle.load(handle)
                if obj_id not in meta["object_ids"]:
                    continue
                obj_idx = list(meta["object_ids"]).index(obj_id)
                centroid = batch["centroid"][i].numpy()
                observed_points = batch["points"][i].numpy()[:, :3] + centroid
                samples.append({
                    "pred_R": pred_Rs[i], "pred_t": centroid + pred_t_residuals[i],
                    "obs_pts": observed_points,
                    "icp_obs_pts": observed_points,
                    "obj_name": meta["object_names"][obj_idx], "obj_id": obj_id,
                    "scale": batch["scale"][i].numpy(), "objects_df": objects_df,
                    "icp_stages": args.icp_stages, "icp_threshold": args.icp_threshold,
                    "max_rot_change": args.max_rot_change, "rgb_path": rgb_path,
                    "use_quarter_turn_fallback": quarter_turn_fallback,
                    "obj_dims": batch["obj_dims"][i].numpy(), "K": batch["intrinsic"][i].numpy(),
                })
    print(f"{initializer} predicted {len(samples)} objects from {len(scene_names)} test scene(s) on {device}.")
    if refine_with_icp and samples:
        icp.run_icp_refinement(samples, use_icp=True)
    return samples


def load_scene_names(scene_list_path):
    """Load one scene identifier per line from a split/list file."""
    try:
        with open(scene_list_path, "r", encoding="utf-8-sig") as handle:
            return {line.strip() for line in handle if line.strip()}
    except UnicodeDecodeError:
        with open(scene_list_path, "r", encoding="utf-16") as handle:
            return {line.strip() for line in handle if line.strip()}


def run_labelled_inference(split="val", max_scenes=None, scene_prefix=None,
                           scene_names=None, checkpoint_path=DEFAULT_CHECKPOINT,
                           refine_with_icp=True, quarter_turn_fallback=False,
                           model_name="pointnet"):
    """Evaluate the selected initializer, with optional ICP, against labelled poses.

    ``scene_names`` can be a set/list loaded with :func:`load_scene_names`, which is
    useful for evaluating a curated level-2 subset. Train augmentation is disabled so
    the result is a deterministic evaluation rather than an augmented training pass.
    """
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")

    args = config.get_config(args=[])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    objects_df = pd.read_csv(PROJECT_ROOT / args.objects_csv)
    dataset = PoseDataset(
        split, args.training_data_dir, args.split_dir, num_points=args.num_points,
        return_image=model_name == "fusion", image_size=args.image_size,
    )

    # PoseDataset augments only when this attribute is literally 'train'. Its cached
    # samples remain valid after changing it, while __getitem__ becomes evaluation-only.
    if split == "train":
        dataset.split_name = "evaluation"

    allowed_scenes = set(scene_names) if scene_names is not None else None
    selected_scenes, selected_indices = [], []
    for dataset_idx, (scene_idx, _) in enumerate(dataset.samples):
        scene_name = os.path.basename(dataset.rgb_files[scene_idx]).split("_")[0]
        if scene_prefix is not None and not scene_name.startswith(scene_prefix):
            continue
        if allowed_scenes is not None and scene_name not in allowed_scenes:
            continue
        if scene_name not in selected_scenes:
            if max_scenes is not None and len(selected_scenes) >= max_scenes:
                continue
            selected_scenes.append(scene_name)
        selected_indices.append(dataset_idx)

    if not selected_indices:
        raise ValueError("No labelled objects matched the requested scene selection.")

    loader = DataLoader(Subset(dataset, selected_indices),
                        batch_size=min(args.batch_size, len(selected_indices)),
                        shuffle=False, num_workers=0)
    model = build_pose_model(model_name, num_classes=args.num_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    samples = []
    with torch.no_grad():
        initializer = "RGB fusion" if model_name == "fusion" else "PointNet"
        for batch in tqdm(loader, desc=f"{initializer} {split} inference", unit="batch"):
            points = batch["points"].to(device)
            object_ids = batch["obj_id"].to(device)
            rot6d, pred_t_residual, auxiliary = _predict_pose(
                model, batch, points, object_ids, device, model_name
            )
            pred_Rs = utils.rotation_6d_to_matrix(rot6d).cpu().numpy()
            pred_t_residuals = pred_t_residual.cpu().numpy()

            for i, rgb_path in enumerate(batch["rgb_path"]):
                if batch["valid_points"][i].item() < args.min_valid_points:
                    continue
                obj_id = batch["obj_id"][i].item()
                with open(rgb_path.replace("_color_kinect.png", "_meta.pkl"), "rb") as handle:
                    meta = pickle.load(handle)
                obj_idx = list(meta["object_ids"]).index(obj_id)
                obj_name = meta["object_names"][obj_idx]
                centroid = batch["centroid"][i].numpy()
                pred_t = centroid + pred_t_residuals[i]
                observed_points = batch["points"][i].numpy()[:, :3] + centroid
                samples.append({
                    "pred_R": pred_Rs[i], "pred_t": pred_t,
                    "pointnet_R": pred_Rs[i].copy(), "pointnet_t": pred_t.copy(),
                    "obs_pts": observed_points,
                    "icp_obs_pts": observed_points,
                    "obj_name": obj_name, "obj_id": obj_id,
                    "scale": batch["scale"][i].numpy(), "objects_df": objects_df,
                    "icp_stages": args.icp_stages, "icp_threshold": args.icp_threshold,
                    "max_rot_change": args.max_rot_change,
                    "use_quarter_turn_fallback": quarter_turn_fallback,
                    "gt_R": batch["gt_rot"][i].numpy(),
                    "gt_t": centroid + batch["gt_t_residual"][i].numpy(),
                    "valid_points": batch["valid_points"][i].item(),
                    "sym": pose_losses.get_object_symmetry(obj_name),
                    "rgb_path": rgb_path, "obj_dims": batch["obj_dims"][i].numpy(),
                    "K": batch["intrinsic"][i].numpy(),
                })

    if refine_with_icp and samples:
        icp.run_icp_refinement(samples, use_icp=True)

    summary = summarize_labelled_predictions(samples, model_name=model_name)
    print(f"Evaluated {len(samples)} objects from {len(selected_scenes)} {split} scene(s) on {device}.")
    return samples, summary


def summarize_labelled_predictions(samples, model_name="pointnet"):
    """Print and return initializer-only and initializer+ICP GT pose errors."""
    if not samples:
        raise ValueError("No samples were available to summarize.")

    pointnet_rot, pointnet_trans, final_rot, final_trans = [], [], [], []
    for sample in samples:
        pointnet_rot.append(pose_losses.compute_symmetry_aware_loss(
            sample["pointnet_R"], sample["gt_R"], sample["sym"]))
        pointnet_trans.append(np.linalg.norm(sample["pointnet_t"] - sample["gt_t"]) * 100)
        final_rot.append(pose_losses.compute_symmetry_aware_loss(
            sample["pred_R"], sample["gt_R"], sample["sym"]))
        final_trans.append(np.linalg.norm(sample["pred_t"] - sample["gt_t"]) * 100)

    summary = {
        "pointnet": {"mean_rot_deg": float(np.mean(pointnet_rot)), "mean_trans_cm": float(np.mean(pointnet_trans))},
        "pointnet_icp": {"mean_rot_deg": float(np.mean(final_rot)), "mean_trans_cm": float(np.mean(final_trans))},
    }
    initializer = "RGB fusion" if model_name == "fusion" else "PointNet"
    print(f"{initializer}:       mean rotation {summary['pointnet']['mean_rot_deg']:.2f} deg | "
          f"mean translation {summary['pointnet']['mean_trans_cm']:.2f} cm")
    print(f"{initializer} + ICP: mean rotation {summary['pointnet_icp']['mean_rot_deg']:.2f} deg | "
          f"mean translation {summary['pointnet_icp']['mean_trans_cm']:.2f} cm")
    return summary


"""Compact reusable workflows for labelled pose-evaluation experiments."""

from dataclasses import dataclass

import helpers as error_analysis




@dataclass
class EvaluationResult:
    """All artifacts from one reproducible labelled evaluation."""

    samples: list
    metrics: dict
    report: object
    overall: object
    by_object: object
    by_scene: object
    csv_path: object
    label: str


def evaluate_validation(checkpoint_path, *, quarter_turn_fallback=False,
                        output_path=None, model_name="pointnet"):
    """Evaluate a checkpoint once and build its instance/object/scene reports."""
    label = "quarter-turn ICP fallback" if quarter_turn_fallback else "standard guarded ICP"
    if output_path is None:
        suffix = "quarter_turn_icp" if quarter_turn_fallback else "instance_error"
        output_path = f"outputs/val_{model_name}_{suffix}_report.csv"

    samples, metrics = run_labelled_inference(
        split="val",
        checkpoint_path=checkpoint_path,
        refine_with_icp=True,
        quarter_turn_fallback=quarter_turn_fallback,
        model_name=model_name,
    )
    report = error_analysis.build_error_report(samples)
    overall, by_object, by_scene = error_analysis.summarize_error_report(report)
    csv_path = error_analysis.export_error_report(report, output_path)
    print(f"Saved {label} report to: {csv_path}")
    return EvaluationResult(
        samples=samples, metrics=metrics, report=report,
        overall=overall, by_object=by_object, by_scene=by_scene,
        csv_path=csv_path, label=label,
    )


def display_summary(result, top_n=10, show_details=False, show_fallback_rows=False):
    """Print key metrics; detailed tables remain available on demand and in CSV."""
    from IPython.display import display

    metrics = result.overall.iloc[0]
    print(
        f"{result.label}: {int(metrics['instances'])} instances | "
        f"rotation {metrics['mean_final_rot_deg']:.2f}° mean / "
        f"{metrics['median_final_rot_deg']:.2f}° median | "
        f"translation {metrics['mean_final_trans_cm']:.2f} cm mean | "
        f"success {metrics['pass_rate_5deg_1cm']:.1f}%"
    )
    print(f"Detailed instance, object, and scene results: {result.csv_path}")
    if show_details:
        display(result.by_object.head(top_n))
        display(result.by_scene.head(top_n))
    if show_fallback_rows:
        display(result.report[result.report["icp_quarter_turn_used"]].head(top_n))


def visualize_hard_scenes(result, max_scenes=5):
    """Show GT, PointNet, and final-pose overlays for one evaluation result."""
    print(f"Visualizing {result.label} results.")
    import visualization
    visualization.visualize_worst_validation_scenes(
        result.samples, result.by_scene, max_scenes=max_scenes,
    )


"""Compare the main PointNet / ICP / RGB--point-fusion ablations fairly.

Each learned model is evaluated once on the full labelled validation split using
the same random point sampling seed.  Its pre-ICP and guarded-ICP values are
then derived from that same inference pass, avoiding an unnecessary second
network run and keeping the four reported configurations directly comparable.
"""

import argparse
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import helpers as error_analysis



PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_POINTNET = PROJECT_ROOT / "model_weights" / "pointnet_occlusion_guarded_v1.pth"
DEFAULT_FUSION = PROJECT_ROOT / "model_weights" / "point_image_fusion_v1.pth"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def summarize(report, name, pose_prefix):
    if pose_prefix == "pointnet":
        rot, trans = report["pointnet_rot_deg"], report["pointnet_trans_cm"]
    else:
        rot, trans = report["final_rot_deg"], report["final_trans_cm"]
    success = ((rot <= 5.0) & (trans <= 1.0)).mean() * 100.0
    return {
        "method": name,
        "instances": len(report),
        "mean_rotation_deg": float(rot.mean()),
        "median_rotation_deg": float(rot.median()),
        "mean_translation_cm": float(trans.mean()),
        "median_translation_cm": float(trans.median()),
        "success_5deg_1cm_pct": float(success),
    }


def evaluate_model(checkpoint, model_name, seed, per_instance_path):
    set_seed(seed)
    samples, _ = run_labelled_inference(
        split="val", checkpoint_path=str(checkpoint), model_name=model_name,
        refine_with_icp=True,
    )
    report = error_analysis.build_error_report(samples)
    error_analysis.export_error_report(report, str(per_instance_path))
    return report


def make_chart(table, path):
    labels = table["method"].tolist()
    figures = [
        ("mean_rotation_deg", "Mean rotation error (deg)", "#4C78A8"),
        ("mean_translation_cm", "Mean translation error (cm)", "#F58518"),
        ("success_5deg_1cm_pct", "Success rate (%)", "#54A24B"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for axis, (column, title, color) in zip(axes, figures):
        bars = axis.bar(labels, table[column], color=color)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)
        for bar, value in zip(bars, table[column]):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}",
                      ha="center", va="bottom", fontsize=9)
        axis.margins(y=0.16)
    fig.suptitle("Validation ablation comparison", fontsize=14)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_ablation_comparison():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointnet_checkpoint", default=str(DEFAULT_POINTNET))
    parser.add_argument("--fusion_checkpoint", default=str(DEFAULT_FUSION))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for checkpoint in (args.pointnet_checkpoint, args.fusion_checkpoint):
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    pointnet_report = evaluate_model(
        args.pointnet_checkpoint, "pointnet", args.seed,
        output_dir / "ablation_pointnet_instances.csv",
    )
    fusion_report = evaluate_model(
        args.fusion_checkpoint, "fusion", args.seed,
        output_dir / "ablation_rgb_point_fusion_instances.csv",
    )
    table = pd.DataFrame([
        summarize(pointnet_report, "PointNet", "pointnet"),
        summarize(pointnet_report, "PointNet + guarded ICP", "final"),
        summarize(fusion_report, "RGB-point fusion", "pointnet"),
        summarize(fusion_report, "RGB-point fusion + guarded ICP", "final"),
    ])
    csv_path = output_dir / "main_ablation_summary.csv"
    chart_path = output_dir / "main_ablation_comparison.png"
    table.to_csv(csv_path, index=False)
    make_chart(table, chart_path)
    print("\nMain validation ablation comparison:")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSummary: {csv_path}")
    print(f"Chart:   {chart_path}")

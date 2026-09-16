"""PointNet + ICP inference and visualization for the unlabelled test set."""

from pathlib import Path
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
# Use the stable standard renderer instead of the VS Code notebook widget.
from tqdm import tqdm

import config
import icp
import loss as loss_utils
import utils
from data import PoseDataset
from model import PointNetBaseline

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TEST_DATA_DIR = Path(r"C:\Users\Owner\Desktop\Fall2023\CSE275\6D Pose\testing_data_pose_filtered\testing_data\v2.2")
DEFAULT_TEST_SPLIT_DIR = PROJECT_ROOT / "training_data_filtered" / "training_data" / "splits" / "v2"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "model_weights" / "pointnet_v2.pth"


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
                       quarter_turn_fallback=False):
    """Predict test poses with PointNet and optional ICP, without GT metrics."""
    args = config.get_config(args=[])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    objects_df = pd.read_csv(PROJECT_ROOT / args.objects_csv)
    dataset = PoseDataset("test", str(test_data_dir), str(test_split_dir), num_points=args.num_points)
    indices, scene_names = _scene_subset(dataset, max_scenes, scene_prefix=scene_prefix)
    if not indices:
        prefix_text = repr(scene_prefix) if scene_prefix is not None else "the requested selection"
        raise ValueError(f"No test objects found for {prefix_text}.")
    loader = DataLoader(Subset(dataset, indices), batch_size=min(args.batch_size, len(indices)),
                        shuffle=False, num_workers=0)

    model = PointNetBaseline(num_classes=args.num_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    samples = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="PointNet test inference", unit="batch"):
            points = batch["points"].to(device)
            rot6d, pred_t_residual = model(points, batch["obj_id"].to(device))
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
                samples.append({
                    "pred_R": pred_Rs[i], "pred_t": centroid + pred_t_residuals[i],
                    "obs_pts": batch["points"][i].numpy()[:, :3] + centroid,
                    "obj_name": meta["object_names"][obj_idx], "obj_id": obj_id,
                    "scale": batch["scale"][i].numpy(), "objects_df": objects_df,
                    "icp_stages": args.icp_stages, "icp_threshold": args.icp_threshold,
                    "max_rot_change": args.max_rot_change, "rgb_path": rgb_path,
                    "use_quarter_turn_fallback": quarter_turn_fallback,
                    "obj_dims": batch["obj_dims"][i].numpy(), "K": batch["intrinsic"][i].numpy(),
                })
    print(f"PointNet predicted {len(samples)} objects from {len(scene_names)} test scene(s) on {device}.")
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
                           refine_with_icp=True, quarter_turn_fallback=False):
    """Evaluate PointNet, with optional ICP, against GT poses on train or validation data.

    ``scene_names`` can be a set/list loaded with :func:`load_scene_names`, which is
    useful for evaluating a curated level-2 subset. Train augmentation is disabled so
    the result is a deterministic evaluation rather than an augmented training pass.
    """
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")

    args = config.get_config(args=[])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    objects_df = pd.read_csv(PROJECT_ROOT / args.objects_csv)
    dataset = PoseDataset(split, args.training_data_dir, args.split_dir, num_points=args.num_points)

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
    model = PointNetBaseline(num_classes=args.num_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    samples = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"PointNet {split} inference", unit="batch"):
            points = batch["points"].to(device)
            rot6d, pred_t_residual = model(points, batch["obj_id"].to(device))
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
                samples.append({
                    "pred_R": pred_Rs[i], "pred_t": pred_t,
                    "pointnet_R": pred_Rs[i].copy(), "pointnet_t": pred_t.copy(),
                    "obs_pts": batch["points"][i].numpy()[:, :3] + centroid,
                    "obj_name": obj_name, "obj_id": obj_id,
                    "scale": batch["scale"][i].numpy(), "objects_df": objects_df,
                    "icp_stages": args.icp_stages, "icp_threshold": args.icp_threshold,
                    "max_rot_change": args.max_rot_change,
                    "use_quarter_turn_fallback": quarter_turn_fallback,
                    "gt_R": batch["gt_rot"][i].numpy(),
                    "gt_t": centroid + batch["gt_t_residual"][i].numpy(),
                    "valid_points": batch["valid_points"][i].item(),
                    "sym": loss_utils.get_object_symmetry(obj_name),
                    "rgb_path": rgb_path, "obj_dims": batch["obj_dims"][i].numpy(),
                    "K": batch["intrinsic"][i].numpy(),
                })

    if refine_with_icp and samples:
        icp.run_icp_refinement(samples, use_icp=True)

    summary = summarize_labelled_predictions(samples)
    print(f"Evaluated {len(samples)} objects from {len(selected_scenes)} {split} scene(s) on {device}.")
    return samples, summary


def summarize_labelled_predictions(samples):
    """Print and return mean PointNet-only and PointNet+ICP GT pose errors."""
    if not samples:
        raise ValueError("No samples were available to summarize.")

    pointnet_rot, pointnet_trans, final_rot, final_trans = [], [], [], []
    for sample in samples:
        pointnet_rot.append(loss_utils.compute_symmetry_aware_loss(
            sample["pointnet_R"], sample["gt_R"], sample["sym"]))
        pointnet_trans.append(np.linalg.norm(sample["pointnet_t"] - sample["gt_t"]) * 100)
        final_rot.append(loss_utils.compute_symmetry_aware_loss(
            sample["pred_R"], sample["gt_R"], sample["sym"]))
        final_trans.append(np.linalg.norm(sample["pred_t"] - sample["gt_t"]) * 100)

    summary = {
        "pointnet": {"mean_rot_deg": float(np.mean(pointnet_rot)), "mean_trans_cm": float(np.mean(pointnet_trans))},
        "pointnet_icp": {"mean_rot_deg": float(np.mean(final_rot)), "mean_trans_cm": float(np.mean(final_trans))},
    }
    print(f"PointNet:       mean rotation {summary['pointnet']['mean_rot_deg']:.2f} deg | "
          f"mean translation {summary['pointnet']['mean_trans_cm']:.2f} cm")
    print(f"PointNet + ICP: mean rotation {summary['pointnet_icp']['mean_rot_deg']:.2f} deg | "
          f"mean translation {summary['pointnet_icp']['mean_trans_cm']:.2f} cm")
    return summary

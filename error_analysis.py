"""Per-object-instance error reports for labelled PointNet + ICP evaluation."""

from pathlib import Path
import os

import numpy as np
import pandas as pd

import loss as loss_utils


def build_error_report(samples, rotation_threshold_deg=5.0, translation_threshold_cm=1.0):
    """Return one diagnostic row per object instance, ordered from worst to best."""
    rows = []
    for sample in samples:
        pointnet_rot = loss_utils.compute_symmetry_aware_loss(sample["pointnet_R"], sample["gt_R"], sample["sym"])
        pointnet_trans = np.linalg.norm(sample["pointnet_t"] - sample["gt_t"]) * 100
        final_rot = loss_utils.compute_symmetry_aware_loss(sample["pred_R"], sample["gt_R"], sample["sym"])
        final_trans = np.linalg.norm(sample["pred_t"] - sample["gt_t"]) * 100
        final_pass = final_rot <= rotation_threshold_deg and final_trans <= translation_threshold_cm
        rows.append({
            "scene": os.path.basename(sample["rgb_path"]).split("_")[0],
            "object_name": sample["obj_name"], "object_id": sample["obj_id"],
            "symmetry": sample["sym"], "valid_points": sample.get("valid_points", np.nan),
            "pointnet_rot_deg": pointnet_rot, "pointnet_trans_cm": pointnet_trans,
            "final_rot_deg": final_rot, "final_trans_cm": final_trans,
            "icp_rot_improvement_deg": pointnet_rot - final_rot,
            "icp_trans_improvement_cm": pointnet_trans - final_trans,
            "icp_fitness": sample.get("icp_fitness", np.nan),
            "icp_rmse_cm": sample.get("icp_rmse", np.nan) * 100,
            "icp_quarter_turn_used": sample.get("icp_quarter_turn_used", False),
            "final_pass_5deg_1cm": final_pass, "rgb_path": sample["rgb_path"],
        })
    report = pd.DataFrame(rows)
    if report.empty:
        return report
    return report.sort_values(["final_pass_5deg_1cm", "final_rot_deg", "final_trans_cm"],
                              ascending=[True, False, False]).reset_index(drop=True)


def summarize_error_report(report):
    """Return overall, per-object, and per-scene summaries for an instance report."""
    if report.empty:
        raise ValueError("The error report is empty.")
    overall = pd.DataFrame([{
        "instances": len(report),
        "mean_final_rot_deg": report["final_rot_deg"].mean(),
        "median_final_rot_deg": report["final_rot_deg"].median(),
        "mean_final_trans_cm": report["final_trans_cm"].mean(),
        "median_final_trans_cm": report["final_trans_cm"].median(),
        "pass_rate_5deg_1cm": report["final_pass_5deg_1cm"].mean() * 100,
        "quarter_turn_fallbacks": report["icp_quarter_turn_used"].sum(),
    }])
    by_object = report.groupby("object_name", as_index=False).agg(
        instances=("object_id", "count"), mean_final_rot_deg=("final_rot_deg", "mean"),
        mean_final_trans_cm=("final_trans_cm", "mean"), mean_valid_points=("valid_points", "mean"),
        mean_icp_fitness=("icp_fitness", "mean"), pass_rate=("final_pass_5deg_1cm", "mean"),
        quarter_turn_fallbacks=("icp_quarter_turn_used", "sum"),
    )
    by_object["pass_rate"] *= 100
    by_object = by_object.sort_values("mean_final_rot_deg", ascending=False).reset_index(drop=True)
    by_scene = report.groupby("scene", as_index=False).agg(
        instances=("object_id", "count"), mean_final_rot_deg=("final_rot_deg", "mean"),
        mean_final_trans_cm=("final_trans_cm", "mean"), mean_valid_points=("valid_points", "mean"),
        mean_icp_fitness=("icp_fitness", "mean"), pass_rate=("final_pass_5deg_1cm", "mean"),
        quarter_turn_fallbacks=("icp_quarter_turn_used", "sum"),
    )
    by_scene["pass_rate"] *= 100
    by_scene = by_scene.sort_values("mean_final_rot_deg", ascending=False).reset_index(drop=True)
    return overall, by_object, by_scene


def export_error_report(report, output_path):
    """Write the per-instance report to CSV and return its resolved path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    return output_path.resolve()

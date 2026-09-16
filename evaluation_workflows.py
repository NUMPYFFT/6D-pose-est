"""Compact reusable workflows for labelled pose-evaluation experiments."""

from dataclasses import dataclass

import error_analysis
import pose_visualization
import test_inference


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
                        output_path=None):
    """Evaluate a checkpoint once and build its instance/object/scene reports."""
    label = "quarter-turn ICP fallback" if quarter_turn_fallback else "standard guarded ICP"
    if output_path is None:
        output_path = (
            "outputs/val_quarter_turn_icp_report.csv"
            if quarter_turn_fallback else "outputs/val_instance_error_report.csv"
        )

    samples, metrics = test_inference.run_labelled_inference(
        split="val",
        checkpoint_path=checkpoint_path,
        refine_with_icp=True,
        quarter_turn_fallback=quarter_turn_fallback,
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


def display_summary(result, top_n=20, show_fallback_rows=False):
    """Display the compact report views used in the notebook."""
    from IPython.display import display

    print(f"Results: {result.label}")
    display(result.overall)
    display(result.by_object.head(top_n))
    display(result.by_scene.head(top_n))
    if show_fallback_rows:
        display(result.report[result.report["icp_quarter_turn_used"]].head(top_n))


def visualize_hard_scenes(result, max_scenes=5):
    """Show GT, PointNet, and final-pose overlays for one evaluation result."""
    print(f"Visualizing {result.label} results.")
    pose_visualization.visualize_worst_validation_scenes(
        result.samples, result.by_scene, max_scenes=max_scenes,
    )

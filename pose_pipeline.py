"""Single command-line entry point for the final 6D pose-estimation pipeline.

Examples
--------
python pose_pipeline.py train --epochs 400
python pose_pipeline.py evaluate
python pose_pipeline.py ablations
python pose_pipeline.py visualize-failure
python pose_pipeline.py infer-test
"""

import argparse
import sys

import inference
import visualization
import preprocess
import helpers

from train import train


BEST_CHECKPOINT = "model_weights/pointnet_occlusion_guarded_v1.pth"
BEST_CHECKPOINT = "model_weights/pointnet_occlusion_guarded_nightly_ft_v2.pth"


def main():
    if sys.argv[1:3] == ["train", "--help"]:
        original_argv = sys.argv
        try:
            sys.argv = ["train.py", "--help"]
            train()
        finally:
            sys.argv = original_argv
        return

    parser = argparse.ArgumentParser(description="6D pose-estimation pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Train the PointNet baseline; accepts train.py options")
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate PointNet + guarded ICP on validation")
    eval_parser.add_argument("--checkpoint", default=BEST_CHECKPOINT)
    subparsers.add_parser("ablations", help="Run the four-way full-validation ablation")
    cache_parser = subparsers.add_parser("preprocess", help="Build point-cloud samples for a split")
    cache_parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    fast_cache_parser = subparsers.add_parser("build-cache", help="Pack preprocessed samples into a memory-mapped cache")
    fast_cache_parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    failure_parser = subparsers.add_parser("visualize-failure", help="Render a four-method same-object comparison")
    failure_parser.add_argument("--scene", default="2-6-3")
    failure_parser.add_argument("--object", dest="object_name", default="mustard_bottle")
    improve_parser = subparsers.add_parser("visualize-icp", help="Render representative ICP improvements")
    improve_parser.add_argument("--model", choices=["pointnet", "fusion"], default="pointnet")
    test_parser = subparsers.add_parser("infer-test", help="Run qualitative test-set inference")
    test_parser.add_argument("--max-scenes", type=int, default=5)
    args, extra = parser.parse_known_args()
    if args.command != "train" and extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")

    if args.command == "train":
        original_argv = sys.argv
        try:
            sys.argv = ["train.py", *extra]
            train()
        finally:
            sys.argv = original_argv
    elif args.command == "evaluate":
        result = inference.evaluate_validation(args.checkpoint)
        inference.display_summary(result)
    elif args.command == "ablations":
        original_argv = sys.argv
        try:
            sys.argv = ["compare_main_ablations.py"]
            inference.run_ablation_comparison()
        finally:
            sys.argv = original_argv
    elif args.command in {"preprocess", "build-cache"}:
        original_argv = sys.argv
        try:
            sys.argv = ["preprocess.py", "--split", args.split]
            if args.command == "preprocess":
                preprocess.preprocess_dataset(helpers.get_config())
            else:
                preprocess.build_fast_cache_main()
        finally:
            sys.argv = original_argv
    elif args.command == "visualize-failure":
        original_argv = sys.argv
        try:
            sys.argv = ["visualize_main_ablations.py", "--scene", args.scene,
                        "--object", args.object_name]
            visualization.visualize_main_ablations()
        finally:
            sys.argv = original_argv
    elif args.command == "visualize-icp":
        original_argv = sys.argv
        try:
            sys.argv = ["visualize_icp_improvements.py", "--model", args.model]
            visualization.visualize_icp_improvements()
        finally:
            sys.argv = original_argv
    else:
        samples = inference.run_test_inference(max_scenes=args.max_scenes)
        print(f"Generated {len(samples)} qualitative test predictions.")


if __name__ == "__main__":
    main()

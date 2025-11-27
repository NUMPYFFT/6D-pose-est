import argparse

def get_config(args=None):
    parser = argparse.ArgumentParser(description="6D Pose Estimation Configuration")

    # Model
    # parser.add_argument("--model", type=str, default="pointnet", choices=["pointnet"], help="Model architecture")
    parser.add_argument("--num_points", type=int, default=1024, help="Number of points to sample")
    parser.add_argument("--num_classes", type=int, default=79, help="Number of object classes")

    # Training
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loading workers")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=400, help="Number of epochs")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    
    # Loss Weights
    parser.add_argument("--w_pm", type=float, default=1.0, help="Weight for Point Matching Loss")
    parser.add_argument("--w_geo", type=float, default=1.0, help="Weight for Geodesic Loss")
    parser.add_argument("--w_trans", type=float, default=1.0, help="Weight for Translation Loss")

    # Evaluation / ICP
    parser.add_argument("--no_icp", action="store_true", help="Disable ICP refinement")
    parser.add_argument("--icp_stages", type=int, default=3, help="Number of ICP stages")
    parser.add_argument("--icp_threshold", type=float, default=0.01, help="Coarse ICP threshold")
    parser.add_argument("--max_rot_change", type=float, default=10.0, help="Max allowed rotation change in degrees")
    parser.add_argument("--min_valid_points", type=int, default=50, help="Minimum number of valid points required to evaluate an object")
    parser.add_argument("--split", type=str, default="val", help="Split to evaluate on (val or test)")

    # Paths
    parser.add_argument("--training_data_dir", type=str, default="./training_data_filtered/training_data/v2.2", help="Path to training data")
    parser.add_argument("--split_dir", type=str, default="./training_data_filtered/training_data/splits/v2", help="Path to split files")
    parser.add_argument("--objects_csv", type=str, default="models/objects_v1.csv", help="Path to objects CSV")
    parser.add_argument("--checkpoint_path", type=str, default="pointnet_v2.pth", help="Path to save checkpoint")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to load checkpoint from (if different from save path)")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--output_dir", type=str, default="output_images", help="Directory for output images")

    return parser.parse_args(args)

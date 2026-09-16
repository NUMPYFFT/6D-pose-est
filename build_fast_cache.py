"""Command-line builder for the optional memory-mapped data cache."""

import config
from data import PoseDataset
from fast_cache import build_mmap_cache


def main():
    args = config.get_config()
    dataset = PoseDataset(args.split, args.training_data_dir, args.split_dir,
                          num_points=args.num_points)
    build_mmap_cache(args.training_data_dir, args.split, expected_samples=len(dataset))


if __name__ == "__main__":
    main()

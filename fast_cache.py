"""Build and validate shared memory-mapped point-cloud caches."""

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


CACHE_VERSION = 1
ARRAY_SPECS = {
    "points": (np.float32, (3,)), "colors": (np.float32, (3,)),
    "normals": (np.float32, (3,)), "gt_R": (np.float32, (3, 3)),
    "gt_t_residual": (np.float32, (3,)), "centroid": (np.float32, (3,)),
    "obj_id": (np.int64, ()), "sym": ("<U64", ()),
    "dims": (np.float32, (3,)), "scale": (np.float32, (3,)),
    "K": (np.float32, (3, 3)), "rgb_path": ("<U512", ()),
}


def mmap_cache_dir(data_dir, split):
    return Path(data_dir) / "preprocessed" / f"{split}_mmap"


def mmap_cache_is_valid(data_dir, split, expected_samples):
    cache_dir = mmap_cache_dir(data_dir, split)
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return (manifest.get("version") == CACHE_VERSION
                and manifest.get("sample_count") == expected_samples
                and all((cache_dir / f"{name}.npy").exists()
                        for name in (*ARRAY_SPECS, "offsets")))
    except (OSError, ValueError, TypeError):
        return False


def build_mmap_cache(data_dir, split, expected_samples=None):
    """Pack object NPZ files into file-backed arrays shared by DataLoader workers."""
    source_dir = Path(data_dir) / "preprocessed" / split
    files = sorted(source_dir.glob("*.npz"))
    if expected_samples is not None and len(files) != expected_samples:
        raise RuntimeError(f"Expected {expected_samples} samples but found {len(files)} NPZ files.")
    if not files:
        raise FileNotFoundError(f"No preprocessed NPZ files found in {source_dir}")

    target_dir = mmap_cache_dir(data_dir, split)
    if target_dir.exists():
        if mmap_cache_is_valid(data_dir, split, len(files)):
            print(f"Memory-mapped {split} cache already exists: {target_dir}")
            return target_dir
        raise FileExistsError(f"Incomplete cache exists at {target_dir}; rename or remove it before rebuilding.")
    target_dir.mkdir(parents=True)

    lengths = np.empty(len(files), dtype=np.int64)
    for index, path in enumerate(tqdm(files, desc=f"Indexing {split} cache", unit="file")):
        with np.load(path) as sample:
            lengths[index] = len(sample["points"])
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    np.save(target_dir / "offsets.npy", offsets)

    arrays = {}
    point_arrays = {"points", "colors", "normals"}
    for name, (dtype, trailing_shape) in ARRAY_SPECS.items():
        shape = ((int(offsets[-1]), *trailing_shape) if name in point_arrays
                 else (len(files), *trailing_shape))
        arrays[name] = np.lib.format.open_memmap(
            target_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
        )

    for index, path in enumerate(tqdm(files, desc=f"Packing {split} cache", unit="file")):
        start, end = offsets[index:index + 2]
        with np.load(path) as sample:
            for name in point_arrays:
                arrays[name][start:end] = sample[name]
            for name in ("gt_R", "gt_t_residual", "centroid", "dims", "scale", "K"):
                arrays[name][index] = sample[name]
            arrays["obj_id"][index] = sample["obj_id"]
            arrays["sym"][index] = str(sample["sym"])
            arrays["rgb_path"][index] = str(sample["rgb_path"])

    for array in arrays.values():
        array.flush()
    (target_dir / "manifest.json").write_text(json.dumps({
        "version": CACHE_VERSION, "sample_count": len(files),
        "total_points": int(offsets[-1]), "source_dir": str(source_dir),
    }, indent=2), encoding="utf-8")
    print(f"Memory-mapped {split} cache ready: {target_dir} ({offsets[-1]:,} points)")
    return target_dir

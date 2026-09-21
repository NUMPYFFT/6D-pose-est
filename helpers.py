import argparse

def get_config(args=None):
    parser = argparse.ArgumentParser(description="6D Pose Estimation Configuration")

    # Model
    parser.add_argument("--model", type=str, default="pointnet", choices=["pointnet", "confidence", "fusion", "correspondence", "multihypothesis"], help="Model architecture")
    parser.add_argument("--num_points", type=int, default=1024, help="Number of points to sample")
    parser.add_argument("--num_classes", type=int, default=79, help="Number of object classes")
    parser.add_argument("--image_size", type=int, default=160, help="Masked RGB crop size for the fusion model")
    parser.add_argument("--num_source_points", type=int, default=512, help="Canonical points for correspondence model")

    # Training
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loading workers")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=400, help="Number of epochs")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--init_checkpoint", type=str, default=None,
                        help="Optional checkpoint whose model weights initialize a new training run. "
                             "Optimizer and scheduler are intentionally reset.")
    parser.add_argument("--reset_best_val", action="store_true",
                        help="When fine-tuning, select the best checkpoint from this run rather than comparing to the source checkpoint's validation loss.")
    parser.add_argument("--snapshot_every", type=int, default=0,
                        help="Save a periodic checkpoint every N epochs (0 disables snapshots).")
    parser.add_argument("--hard_indices", type=str, default=None,
                        help="Optional .npy dataset indices to replay more often during training.")
    parser.add_argument("--hard_replay_factor", type=float, default=4.0,
                        help="Relative sampler weight assigned to mined hard examples.")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    
    # Loss Weights
    parser.add_argument("--w_pm", type=float, default=1.0, help="Weight for Point Matching Loss")
    parser.add_argument("--w_geo", type=float, default=1.0, help="Weight for Geodesic Loss")
    parser.add_argument("--w_trans", type=float, default=1.0, help="Weight for Translation Loss")
    parser.add_argument("--w_corr", type=float, default=0.25, help="Auxiliary correspondence-pose loss weight")
    parser.add_argument("--w_conf", type=float, default=0.25, help="Multi-hypothesis confidence classification loss weight")
    parser.add_argument("--w_diversity", type=float, default=0.05, help="Multi-hypothesis rotation diversity loss weight")
    parser.add_argument("--confidence_keep_ratio", type=float, default=0.70,
                        help="Fraction of highest-confidence observed points retained for ICP (confidence model only)")

    # Evaluation / ICP
    parser.add_argument("--no_icp", action="store_true", help="Disable ICP refinement")
    parser.add_argument("--icp_stages", type=int, default=3, help="Number of ICP stages")
    parser.add_argument("--icp_threshold", type=float, default=0.01, help="Coarse ICP threshold")
    parser.add_argument("--max_rot_change", type=float, default=10.0, help="Max allowed rotation change in degrees")
    parser.add_argument("--min_valid_points", type=int, default=50, help="Minimum number of valid points required to evaluate an object")
    parser.add_argument("--split", type=str, default="val", help="Split to evaluate on (val or test)")

    # Paths
    parser.add_argument("--training_data_dir", type=str, default=r"C:\Users\Owner\Desktop\Fall2023\CSE275\6D Pose\training_data_filtered\training_data\v2.2", help="Path to training data")
    parser.add_argument("--split_dir", type=str, default=r"C:\Users\Owner\Desktop\Fall2023\CSE275\6D Pose\training_data_filtered\training_data\splits\v2", help="Path to split files")
    parser.add_argument("--objects_csv", type=str, default="models/objects_v1.csv", help="Path to objects CSV")
    parser.add_argument("--checkpoint_path", type=str, default="model_weights/pointnet_new.pth", help="Path to save checkpoint")
    parser.add_argument("--output_dir", type=str, default="output_images", help="Directory for output images")

    return parser.parse_args(args)


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


"""Cached canonical mesh sampling for source-conditioned pose models."""

from pathlib import Path

import numpy as np
import trimesh


_SOURCE_CACHE = {}


def canonical_points(object_name, objects_df, scale, num_points=512):
    """Return a fixed-size canonical object cloud, scaled as in the dataset.

    The module cache is intentionally process-local: Windows DataLoader workers
    open a mesh at most once for a given object/scale pair and subsequently only
    return inexpensive copies of the sampled points.
    """
    scale = np.asarray(scale, dtype=np.float32).reshape(-1)
    if scale.size == 1:
        scale = np.repeat(scale, 3)
    key = (str(object_name), tuple(np.round(scale, 6)), int(num_points))
    if key not in _SOURCE_CACHE:
        row = objects_df[objects_df["object"] == object_name]
        if row.empty:
            raise KeyError(f"Canonical mesh is not listed for {object_name!r}.")
        mesh_path = Path(str(row.iloc[0]["location"])) / "visual_meshes" / "visual.dae"
        if not mesh_path.exists():
            raise FileNotFoundError(f"Canonical mesh is missing: {mesh_path}")
        mesh = trimesh.load(mesh_path, force="mesh")
        points = mesh.sample(num_points).astype(np.float32) * scale[None, :]
        _SOURCE_CACHE[key] = points
    return _SOURCE_CACHE[key].copy()


import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import pickle
import open3d as o3d
from tqdm import tqdm
import pandas as pd



def get_split_files(split_name, data_dir, split_dir):
    with open(os.path.join(split_dir, f"{split_name}.txt"), 'r') as f:
        prefix = [os.path.join(data_dir, line.strip()) for line in f if line.strip()]
        rgb = [p + "_color_kinect.png" for p in prefix]
        depth = [p + "_depth_kinect.png" for p in prefix]
        label = [p + "_label_kinect.png" for p in prefix]
        meta = [p + "_meta.pkl" for p in prefix]
    return rgb, depth, label, meta


class PoseDataset(Dataset):
    def __init__(self, split_name, data_dir, split_dir, num_points=4096, subset_size=None,
                 return_image=False, image_size=160, return_source=False,
                 num_source_points=512):
        self.split_name = split_name
        self.rgb_files, self.depth_files, self.label_files, self.meta_files = \
            get_split_files(split_name, data_dir, split_dir)
        self.data_dir = data_dir
        self.num_points = num_points
        self.return_image = return_image
        self.image_size = image_size
        self.return_source = return_source
        self.num_source_points = num_source_points

        # Load object info table
        self.objects_df = pd.read_csv("models/objects_v1.csv")

        # Cache object geometry + symmetry
        self.obj_info_cache = {}
        for _, row in self.objects_df.iterrows():
            name = str(row['object']).strip()
            sym = str(row['geometric_symmetry']).lower()

            self.obj_info_cache[name] = {
                'geometric_symmetry': sym,
                'width': row['width'],
                'length': row['length'],
                'height': row['height']
            }

        # Optional: subsample dataset
        if subset_size is not None and subset_size < len(self.rgb_files):
            idxs = np.random.choice(len(self.rgb_files), subset_size, replace=False)
            self.rgb_files = [self.rgb_files[i] for i in idxs]
            self.depth_files = [self.depth_files[i] for i in idxs]
            self.label_files = [self.label_files[i] for i in idxs]
            self.meta_files = [self.meta_files[i] for i in idxs]

        # -----------------------------------------
        # INDEXING: REMOVE OCCLUDED OBJECTS HERE !!!
        # -----------------------------------------
        self.samples = []
        
        # Check for preprocessed data
        self.preprocessed_dir = os.path.join(data_dir, "preprocessed", split_name)
        # Force absolute path check to be sure
        abs_preprocessed_dir = os.path.abspath(self.preprocessed_dir)
        self.use_preprocessed = os.path.exists(abs_preprocessed_dir)
        
        print(f"Checking for preprocessed data at: {abs_preprocessed_dir}")
        print(f"Exists: {self.use_preprocessed}")
        
        if self.use_preprocessed:
            print(f"Using preprocessed data from {self.preprocessed_dir}")
        else:
            print("WARNING: Preprocessed data not found! Training will be SLOW.")

        # Check for cached index (only if not subsampling)
        cache_path = os.path.join(split_dir, f"{split_name}_index.pkl")
        use_cache = (subset_size is None)
        
        if use_cache and os.path.exists(cache_path):
            print(f"Loading cached index from {cache_path}...")
            with open(cache_path, "rb") as f:
                self.samples = pickle.load(f)
        else:
            print("Indexing dataset (removing invisible objects)...")
            for i in tqdm(range(len(self.meta_files))):

                # Load label map once
                label_img = np.asarray(Image.open(self.label_files[i]), dtype=np.int32)

                with open(self.meta_files[i], "rb") as f:
                    meta = pickle.load(f)

                for oid in meta['object_ids']:

                    # Skip objects NOT in label → fully occluded
                    if np.sum(label_img == oid) == 0:
                        continue

                    # Skip if no world pose (unless testing)
                    if self.split_name != 'test':
                        if 'poses_world' not in meta or meta['poses_world'][oid] is None:
                            continue

                    # Valid training sample
                    self.samples.append((i, oid))
            
            if use_cache:
                print(f"Saving index to {cache_path}...")
                with open(cache_path, "wb") as f:
                    pickle.dump(self.samples, f)

        print(f"Final samples: {len(self.samples)} visible objects.")

        # Optional packed cache: workers lazily memory-map the same file-backed
        # arrays instead of opening one compressed NPZ file per requested object.
        self.use_mmap_cache = self.use_preprocessed and mmap_cache_is_valid(
            self.data_dir, split_name, len(self.samples)
        )
        self.mmap_dir = mmap_cache_dir(self.data_dir, split_name)
        self._mmap_arrays = None
        if self.use_mmap_cache:
            print(f"Using shared memory-mapped cache from {self.mmap_dir}")

        self.sample_object_names = None
        if self.return_source:
            # The regular point cache predates source-conditioned models and does
            # not store object names. Resolve the small scene-level mapping once,
            # rather than reopening metadata for every training sample.
            names_by_scene = {}
            self.sample_object_names = []
            for scene_idx, object_id in self.samples:
                if scene_idx not in names_by_scene:
                    with open(self.meta_files[scene_idx], "rb") as handle:
                        meta = pickle.load(handle)
                    names_by_scene[scene_idx] = dict(zip(
                        list(meta["object_ids"]), list(meta["object_names"])
                    ))
                self.sample_object_names.append(names_by_scene[scene_idx][object_id])

    def __len__(self):
        return len(self.samples)

    def _masked_image_crop(self, scene_idx, obj_id, K, camera_points):
        """Return a masked RGB crop and normalized image locations for points.

        Projection is calculated from the centred cloud restored to camera
        coordinates.  The fourth image channel is the instance mask, which
        prevents the CNN from treating surrounding objects as object texture.
        """
        rgb = np.asarray(Image.open(self.rgb_files[scene_idx]).convert("RGB"), dtype=np.uint8)
        labels = np.asarray(Image.open(self.label_files[scene_idx]), dtype=np.int32)
        mask = labels == obj_id
        height, width = mask.shape
        rows, cols = np.where(mask)
        if len(rows) == 0:
            y0, y1, x0, x1 = 0, height, 0, width
        else:
            padding = max(12, int(0.15 * max(rows.max() - rows.min() + 1,
                                              cols.max() - cols.min() + 1)))
            y0, y1 = max(0, rows.min() - padding), min(height, rows.max() + padding + 1)
            x0, x1 = max(0, cols.min() - padding), min(width, cols.max() + padding + 1)

        rgb_crop = rgb[y0:y1, x0:x1]
        mask_crop = mask[y0:y1, x0:x1].astype(np.uint8) * 255
        resampling = getattr(Image, "Resampling", Image)
        rgb_crop = np.asarray(Image.fromarray(rgb_crop).resize(
            (self.image_size, self.image_size), resample=resampling.BILINEAR
        ), dtype=np.float32) / 255.0
        mask_crop = np.asarray(Image.fromarray(mask_crop).resize(
            (self.image_size, self.image_size), resample=resampling.NEAREST
        ), dtype=np.float32)[..., None] / 255.0
        image = np.concatenate([rgb_crop * mask_crop, mask_crop], axis=-1)

        z = np.clip(camera_points[:, 2], 1e-6, None)
        u = K[0, 0] * camera_points[:, 0] / z + K[0, 2]
        v = K[1, 1] * camera_points[:, 1] / z + K[1, 2]
        crop_w = max(x1 - x0 - 1, 1)
        crop_h = max(y1 - y0 - 1, 1)
        pixel_coords = np.stack([
            2.0 * (u - x0) / crop_w - 1.0,
            2.0 * (v - y0) / crop_h - 1.0,
        ], axis=-1).astype(np.float32)
        return image.transpose(2, 0, 1).astype(np.float32), pixel_coords

    def _open_mmap_cache(self):
        """Open maps lazily inside each spawned Windows worker."""
        if self._mmap_arrays is None:
            names = (
                "offsets", "points", "colors", "normals", "gt_R",
                "gt_t_residual", "centroid", "obj_id", "sym", "dims",
                "scale", "K", "rgb_path",
            )
            self._mmap_arrays = {
                name: np.load(self.mmap_dir / f"{name}.npy", mmap_mode="r")
                for name in names
            }
        return self._mmap_arrays

    def __getitem__(self, idx):
        scene_idx, obj_id = self.samples[idx]
        
        if self.use_mmap_cache:
            cache = self._open_mmap_cache()
            start, end = cache["offsets"][idx:idx + 2]
            pts = cache["points"][start:end]
            colors = cache["colors"][start:end]
            normals = cache["normals"][start:end]
            # Metadata is later converted to tensors.  Copy these small arrays
            # so PyTorch never receives a read-only memory-map view.
            gt_R = cache["gt_R"][idx].copy()
            gt_t_residual = cache["gt_t_residual"][idx].copy()
            centroid = cache["centroid"][idx].copy()
            sym = str(cache["sym"][idx])
            dims = cache["dims"][idx].copy()
            scale = cache["scale"][idx].copy()
            K = cache["K"][idx].copy()
            rgb_path = str(cache["rgb_path"][idx])
            valid_points_count = len(pts)

            N = len(pts)
            idxs = np.random.choice(N, self.num_points, replace=N < self.num_points)
            pts = pts[idxs]
            colors = colors[idxs]
            normals = normals[idxs]

        elif self.use_preprocessed:
            # Load from .npz
            # The preprocess script saved files as {i:06d}.npz where i is the index in self.samples
            # Wait, the preprocess script iterated over dataset.samples.
            # So the index 'idx' here corresponds directly to the file name if we assume the order is preserved.
            # Yes, dataset.samples is loaded from the same pickle file.
            
            npz_path = os.path.join(self.preprocessed_dir, f"{idx:06d}.npz")
            data = np.load(npz_path)
            
            pts = data['points']
            colors = data['colors']
            normals = data['normals']
            gt_R = data['gt_R']
            gt_t_residual = data['gt_t_residual']
            centroid = data['centroid']
            # obj_id = data['obj_id'] # Already have this
            sym = str(data['sym'])
            dims = data['dims']
            scale = data['scale']
            K = data['K']
            rgb_path = str(data['rgb_path'])
            
            valid_points_count = len(pts)
            
            # Sample to num_points
            N = len(pts)
            if N >= self.num_points:
                idxs = np.random.choice(N, self.num_points, replace=False)
            else:
                idxs = np.random.choice(N, self.num_points, replace=True)
                
            pts = pts[idxs]
            colors = colors[idxs]
            normals = normals[idxs]
            
        else:
            # Load images
            depth = np.asarray(Image.open(self.depth_files[scene_idx]), dtype=np.float32) / 1000.0
            rgb = np.asarray(Image.open(self.rgb_files[scene_idx]), dtype=np.float32) / 255.0
            label = np.asarray(Image.open(self.label_files[scene_idx]), dtype=np.int32)

            # Load metadata
            with open(self.meta_files[scene_idx], "rb") as f:
                meta = pickle.load(f)

            # intrinsics
            intr = meta['intrinsic']
            if isinstance(intr, dict):
                K = np.array([
                    [intr['fx'], 0, intr['cx']],
                    [0, intr['fy'], intr['cy']],
                    [0, 0, 1]
                ], dtype=np.float32)
            else:
                K = np.array(intr, dtype=np.float32)

            # Mask object pixels
            mask = (label == obj_id)
            vs, us = np.where(mask)

            # ------------------------------------------------
            # At indexing stage we guarantee len(vs) > 0
            # So we NEVER reach empty cases here.
            # ------------------------------------------------

            # Extract depth
            z = depth[vs, us]

            # Filter invalid depth
            valid = (z > 0.001) & (z < 3.0)
            z = z[valid]
            vs = vs[valid]
            us = us[valid]

            # If depth rejects all points, fallback tiny safe case
            if len(z) == 0:
                pts = np.zeros((1,3), dtype=np.float32)
                colors = np.zeros((1,3), dtype=np.float32)
                normals = np.zeros((1,3), dtype=np.float32)
            else:
                # Project to point cloud
                x = (us - K[0,2]) * z / K[0,0]
                y = (vs - K[1,2]) * z / K[1,1]
                pts = np.stack([x, y, z], axis=-1)
                colors = rgb[vs, us]

                # Downsample
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.colors = o3d.utility.Vector3dVector(colors)

                pcd = pcd.voxel_down_sample(0.005)
                # Better normal estimation with radius search and orientation
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=30))
                pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))

                pts = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors)
                normals = np.asarray(pcd.normals)

            # Sample to fixed size
            N = len(pts)
            valid_points_count = N  # Store original valid count before resampling
            if N >= self.num_points:
                idxs = np.random.choice(N, self.num_points, replace=False)
            else:
                idxs = np.random.choice(N, self.num_points, replace=True)

            pts = pts[idxs]
            normals = normals[idxs]
            colors = colors[idxs]

            # Center
            centroid = np.mean(pts, axis=0)
            pts = pts - centroid

            # Ground truth pose
            T_wc = np.array(meta['extrinsic']).reshape(4,4)
            
            if self.split_name == 'test':
                gt_R = np.eye(3, dtype=np.float32)
                gt_t = np.zeros(3, dtype=np.float32)
                gt_t_residual = np.zeros(3, dtype=np.float32)
            else:
                T_ow = np.array(meta['poses_world'][obj_id]).reshape(4,4)
                T_co = T_wc @ T_ow

                gt_R = T_co[:3,:3]
                gt_t = T_co[:3,3]
                gt_t_residual = gt_t - centroid
            
            # Object name logic for sym (only needed if not preprocessed)
            if obj_id in meta['object_ids']:
                idx2 = list(meta['object_ids']).index(obj_id)
                name = meta['object_names'][idx2]
            else:
                name = "unknown"

            if name in self.obj_info_cache:
                info = self.obj_info_cache[name]
                sym = info['geometric_symmetry']
                dims = np.array([info['width'], info['length'], info['height']], dtype=np.float32)
            else:
                sym = "no"
                dims = np.zeros(3, dtype=np.float32)

            # Scaling
            scale = meta['scales'][obj_id]
            if scale is None:
                scale = np.array([1,1,1], dtype=np.float32)
            else:
                scale = np.array(scale, dtype=np.float32)
                if scale.ndim == 0:
                    scale = np.array([scale, scale, scale], dtype=np.float32)
            
            rgb_path = self.rgb_files[scene_idx]

        # Augmentation for training
        if self.split_name == 'train':
            # 1. Partial-view augmentation.  Real test crops can contain only the
            # visible side of an object because of scene occlusion.  Hide one
            # contiguous side of the centred cloud, then re-centre it around the
            # new visible-cloud centroid.  Translation must use that same centroid
            # convention or this augmentation would teach an inconsistent target.
            if np.random.rand() < 0.65:
                view_axis = np.random.randint(0, 3)
                keep_fraction = np.random.uniform(0.55, 0.90)
                cut_value = np.quantile(pts[:, view_axis], 1.0 - keep_fraction)
                keep = pts[:, view_axis] >= cut_value

                # A degenerate crop is not useful; retain the original cloud.
                if np.count_nonzero(keep) >= 64:
                    pts = pts[keep]
                    colors = colors[keep]
                    normals = normals[keep]

                    # Points are currently expressed relative to the original
                    # centroid.  Move the crop to its own centroid and update both
                    # the absolute centroid and the corresponding pose residual.
                    crop_offset = np.mean(pts, axis=0)
                    pts = pts - crop_offset
                    centroid = centroid + crop_offset
                    gt_t_residual = gt_t_residual - crop_offset

                    resample = np.random.choice(len(pts), self.num_points,
                                                replace=len(pts) < self.num_points)
                    pts = pts[resample]
                    colors = colors[resample]
                    normals = normals[resample]

            # 2. Small sensor-noise augmentation.  Keep it after the centroid
            # update so random zero-mean jitter does not alter the pose target.
            pts = pts + np.random.normal(0, 0.002, pts.shape)
            
            # 3. Targeted large-orientation augmentation. Several difficult
            # validation examples contain containers resting on their sides or
            # flipped, while the data is otherwise dominated by upright poses.
            # The camera-axis rotations below are applied consistently to the
            # cloud, normals, rotation target, and translation residual. This
            # deliberately enriches those rare orientations without relying on
            # an unreliable per-object definition of "up" in mesh coordinates.
            laydown_rotation = np.eye(3)
            orientation_draw = np.random.rand()
            if not self.return_image and orientation_draw < 0.50:
                quarter_turn = np.random.choice((-np.pi / 2, np.pi / 2))
                if np.random.rand() < 0.5:
                    laydown_rotation = np.array([
                        [1, 0, 0],
                        [0, np.cos(quarter_turn), -np.sin(quarter_turn)],
                        [0, np.sin(quarter_turn), np.cos(quarter_turn)],
                    ])
                else:
                    laydown_rotation = np.array([
                        [np.cos(quarter_turn), 0, np.sin(quarter_turn)],
                        [0, 1, 0],
                        [-np.sin(quarter_turn), 0, np.cos(quarter_turn)],
                    ])
            elif not self.return_image and orientation_draw < 0.65:
                # A rarer half turn covers flipped poses such as 2-77-11.  Its
                # horizontal axis is chosen at random for the same camera-frame
                # convention as the quarter-turn cases.
                half_turn = np.pi
                if np.random.rand() < 0.5:
                    laydown_rotation = np.array([
                        [1, 0, 0],
                        [0, np.cos(half_turn), -np.sin(half_turn)],
                        [0, np.sin(half_turn), np.cos(half_turn)],
                    ])
                else:
                    laydown_rotation = np.array([
                        [np.cos(half_turn), 0, np.sin(half_turn)],
                        [0, 1, 0],
                        [-np.sin(half_turn), 0, np.cos(half_turn)],
                    ])

            # Image fusion requires real camera projection to remain valid, so
            # only it skips synthetic rigid camera rotations. It retains point
            # resampling and partial-view cropping above.
            rotation_limit = 0.0 if self.return_image else np.pi / 6
            ang_x = np.random.uniform(-rotation_limit, rotation_limit)
            ang_y = np.random.uniform(-rotation_limit, rotation_limit)
            ang_z = np.random.uniform(-rotation_limit, rotation_limit)
            
            Rx = np.array([[1, 0, 0], [0, np.cos(ang_x), -np.sin(ang_x)], [0, np.sin(ang_x), np.cos(ang_x)]])
            Ry = np.array([[np.cos(ang_y), 0, np.sin(ang_y)], [0, 1, 0], [-np.sin(ang_y), 0, np.cos(ang_y)]])
            Rz = np.array([[np.cos(ang_z), -np.sin(ang_z), 0], [np.sin(ang_z), np.cos(ang_z), 0], [0, 0, 1]])
            
            R_aug = laydown_rotation @ Rz @ Ry @ Rx
            
            pts = pts @ R_aug.T
            normals = normals @ R_aug.T
            gt_R = R_aug @ gt_R
            gt_t_residual = R_aug @ gt_t_residual

        # Final 9D input
        input_data = np.concatenate([pts, colors, normals], axis=1)

        result = {
            'points': torch.from_numpy(input_data).float(),
            'gt_rot': torch.from_numpy(gt_R).float(),
            'gt_t_residual': torch.from_numpy(gt_t_residual).float(),
            'centroid': torch.from_numpy(centroid).float(),
            'obj_id': obj_id,
            'sym_str': sym,
            'rgb_path': rgb_path,
            'intrinsic': K,
            'obj_dims': dims,
            'scale': torch.from_numpy(scale),
            'dataset_idx': idx,
            'valid_points': valid_points_count
        }
        if self.return_image:
            image, pixel_coords = self._masked_image_crop(
                scene_idx, obj_id, K, pts + centroid
            )
            result['image'] = torch.from_numpy(image)
            result['pixel_coords'] = torch.from_numpy(pixel_coords)
        if self.return_source:
            result['source_points'] = torch.from_numpy(canonical_points(
                self.sample_object_names[idx], self.objects_df, scale,
                num_points=self.num_source_points,
            ))
        return result


import os
import pickle
import numpy as np
import torch
import cv2
import torch.nn as nn
from PIL import Image
import open3d as o3d
from tqdm import tqdm
import pandas as pd
from collections import Counter


# --- Rotation conversions ---
def rotation_6d_to_matrix(rot_6d):
    batch_size = rot_6d.shape[0]
    a1 = rot_6d[:, :3]
    a2 = rot_6d[:, 3:]
    b1 = nn.functional.normalize(a1, dim=1, eps=1e-6)
    b2 = a2 - torch.sum(b1 * a2, dim=1, keepdim=True) * b1
    b2 = nn.functional.normalize(b2, dim=1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=1)
    rot_matrix = torch.stack([b1, b2, b3], dim=1).transpose(-2, -1)
    return rot_matrix

def draw_projected_box3d(image, center, size, rotation, extrinsic, intrinsic, thickness=2, color=None):
    """
    Draw a projected 3D bounding box on an image.
    This is the original function signature expected by see_data.ipynb
    
    Args:
        image: Input image (numpy array) - modified in place
        center: [3] 3D center position in world coordinates
        size: [3] box dimensions [width, height, depth]  
        rotation: [3, 3] rotation matrix
        extrinsic: [4, 4] camera extrinsic matrix (world to camera)
        intrinsic: [3, 3] camera intrinsic matrix
        thickness: Line thickness
        color: RGB color tuple for the box (auto-generated if None)
        
    Returns:
        None (modifies image in place)
    """
    # Generate 3D bounding box corners in object coordinates
    w, h, d = size[0]/2, size[1]/2, size[2]/2
    corners_obj = np.array([
        [-w, -h, -d],  # 0
        [ w, -h, -d],  # 1  
        [ w,  h, -d],  # 2
        [-w,  h, -d],  # 3
        [-w, -h,  d],  # 4
        [ w, -h,  d],  # 5
        [ w,  h,  d],  # 6
        [-w,  h,  d],  # 7
    ])
    
    # Transform corners to world coordinates
    corners_world = (rotation @ corners_obj.T).T + center
    
    # Add homogeneous coordinate
    corners_world_hom = np.hstack([corners_world, np.ones((8, 1))])
    
    # Transform to camera coordinates
    corners_cam_hom = (extrinsic @ corners_world_hom.T).T
    corners_cam = corners_cam_hom[:, :3]
    
    # Project to 2D
    corners_2d_hom = (intrinsic @ corners_cam.T).T
    
    # Check for valid depth to avoid division by zero or negative depth issues
    if np.any(corners_2d_hom[:, 2] <= 0) or np.any(np.isnan(corners_2d_hom)) or np.any(np.isinf(corners_2d_hom)):
        return

    corners_2d = corners_2d_hom[:, :2] / corners_2d_hom[:, 2:3]
    
    # Check for valid 2D coordinates before casting
    if np.any(np.isnan(corners_2d)) or np.any(np.isinf(corners_2d)):
        return
        
    try:
        corners_2d = corners_2d.astype(int)
    except (ValueError, OverflowError):
        return
    
    # Generate random color if not specified
    if color is None:
        color = tuple(np.random.randint(0, 256, 3).tolist())
    
    # Define the 12 edges of a 3D box
    edges = [
        # Bottom face
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Top face
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Vertical edges
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    
    # Draw each edge
    h_img, w_img = image.shape[:2]
    for edge in edges:
        pt1 = tuple(corners_2d[edge[0]])
        pt2 = tuple(corners_2d[edge[1]])
        
        # Check if points are within reasonable bounds and have positive depth
        # Relaxed bounds check to allow drawing even if partially off-screen
        if (corners_cam[edge[0], 2] > 0 and corners_cam[edge[1], 2] > 0):
             # Clip to image bounds for safety, but try to draw
             # Actually cv2.line handles clipping, we just need to ensure coordinates are not insane (overflow)
             if (-10000 < pt1[0] < 10000 and -10000 < pt1[1] < 10000 and
                 -10000 < pt2[0] < 10000 and -10000 < pt2[1] < 10000):
                cv2.line(image, pt1, pt2, color, thickness)

def check_class_distribution(args):
    print(f"Checking class distribution for split: train")
    print(f"Data Dir: {args.training_data_dir}")
    print(f"Split Dir: {args.split_dir}")

    # Initialize Dataset (this will load/create the index)
    # We use a small num_points just to init, we won't load actual point clouds
    dataset = PoseDataset("train", args.training_data_dir, args.split_dir, num_points=1024)
    
    # Access the internal samples list: [(scene_idx, obj_id), ...]
    samples = dataset.samples
    
    print(f"Total samples: {len(samples)}")
    
    # Count object IDs
    obj_counts = Counter([s[1] for s in samples])
    
    # Load Object Names for better readability
    # objects_df = pd.read_csv(args.objects_csv)
    
    # Let's iterate and collect names
    id_to_name = {}
    
    print("Resolving object names...")
    
    # Optimization: We only need to find one instance of each ID to get its name.
    found_ids = set()
    for i in range(len(dataset.meta_files)):
        if len(found_ids) == len(obj_counts):
            break
            
        with open(dataset.meta_files[i], "rb") as f:
            meta = pickle.load(f)
            
        for idx, oid in enumerate(meta['object_ids']):
            if oid not in id_to_name:
                id_to_name[oid] = meta['object_names'][idx]
                found_ids.add(oid)

    # Print Distribution
    print("\n" + "="*60)
    print(f"{'ID':<5} | {'Object Name':<30} | {'Count':<10} | {'%':<5}")
    print("-" * 60)
    
    sorted_counts = obj_counts.most_common()
    
    for oid, count in sorted_counts:
        name = id_to_name.get(oid, "Unknown")
        percentage = (count / len(samples)) * 100
        print(f"{oid:<5} | {name:<30} | {count:<10} | {percentage:.2f}%")
        
    print("="*60)
    
    # Statistics
    counts = list(obj_counts.values())
    print(f"\nStatistics:")
    print(f"Min samples: {min(counts)} (ID: {sorted_counts[-1][0]} - {id_to_name.get(sorted_counts[-1][0], 'Unknown')})")
    print(f"Max samples: {max(counts)} (ID: {sorted_counts[0][0]} - {id_to_name.get(sorted_counts[0][0], 'Unknown')})")
    print(f"Mean samples: {sum(counts) / len(counts):.2f}")
    print(f"Median samples: {sorted(counts)[len(counts)//2]}")

"""Per-object-instance error reports for labelled PointNet + ICP evaluation."""

from pathlib import Path
import os

import numpy as np
import pandas as pd

import model as loss_utils


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

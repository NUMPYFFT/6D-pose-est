import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import pickle
import open3d as o3d
from tqdm import tqdm
import pandas as pd

training_data_dir = "./training_data_filtered/training_data/v2.2"
split_dir = "./training_data_filtered/training_data/splits/v2"

def get_split_files(split_name):
    with open(os.path.join(split_dir, f"{split_name}.txt"), 'r') as f:
        prefix = [os.path.join(training_data_dir, line.strip()) for line in f if line.strip()]
        rgb = [p + "_color_kinect.png" for p in prefix]
        depth = [p + "_depth_kinect.png" for p in prefix]
        label = [p + "_label_kinect.png" for p in prefix]
        meta = [p + "_meta.pkl" for p in prefix]
    return rgb, depth, label, meta

def project_depth_to_3d(depth, K):
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    h, w = depth.shape
    v, u = np.indices((h, w))

    # NO 0.5 OFFSET — correct YCB-V projection
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.stack([x, y, z], axis=-1)

class PoseDataset(Dataset):
    def __init__(self, split_name, num_points=4096, subset_size=None):
        self.split_name = split_name
        self.rgb_files, self.depth_files, self.label_files, self.meta_files = get_split_files(split_name)
        self.num_points = num_points

        self.objects_df = pd.read_csv("benchmark_utils/objects_v1.csv")

        # Cache object info
        self.obj_info_cache = {}
        for _, row in self.objects_df.iterrows():
            name = str(row['object']).strip()
            sym = str(row['geometric_symmetry']).lower()
            
            # Manual Overrides (Same as in eval.py)
            # if name in ['g_lego_duplo', 'e_lego_duplo', 'nine_hole_peg_test']: sym = 'z2|x2|y2'
            # if name in ['mustard_bottle', 'bleach_cleanser']: sym = 'z2'
            # if name == 'extra_large_clamp': sym = 'x2|z2'
            
            self.obj_info_cache[name] = {
                'geometric_symmetry': sym,
                'width': row['width'],
                'length': row['length'],
                'height': row['height']
            }

        if subset_size is not None and subset_size < len(self.rgb_files):
            idxs = np.random.choice(len(self.rgb_files), subset_size, replace=False)
            self.rgb_files = [self.rgb_files[i] for i in idxs]
            self.depth_files = [self.depth_files[i] for i in idxs]
            self.label_files = [self.label_files[i] for i in idxs]
            self.meta_files = [self.meta_files[i] for i in idxs]

        self.samples = []
        print("Indexing dataset...")
        for i in tqdm(range(len(self.meta_files))):
            with open(self.meta_files[i], "rb") as f:
                meta = pickle.load(f)
            for oid in meta['object_ids']:
                if meta['poses_world'][oid] is not None:
                    self.samples.append((i, oid))

        print(f"Indexed {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scene_idx, obj_id = self.samples[idx]

        depth = np.asarray(Image.open(self.depth_files[scene_idx]), dtype=np.float32) / 1000.0
        rgb = np.asarray(Image.open(self.rgb_files[scene_idx]), dtype=np.float32) / 255.0
        label = np.asarray(Image.open(self.label_files[scene_idx]), dtype=np.int32)

        with open(self.meta_files[scene_idx], "rb") as f:
            meta = pickle.load(f)

        # intrinsics
        intr = meta['intrinsic']
        if isinstance(intr, dict):
            K = np.array([[intr['fx'], 0, intr['cx']],
                          [0, intr['fy'], intr['cy']],
                          [0, 0, 1]], dtype=np.float32)
        else:
            K = np.array(intr, dtype=np.float32)

        mask = (label == obj_id)
        vs, us = np.where(mask)
        
        # Train: Resample if bad
        if self.split_name == 'train' and len(vs) < 50:
            return self.__getitem__(np.random.randint(0, len(self)))
            
        # Val/Test: Handle empty/small
        if len(vs) == 0:
            input_data = np.zeros((self.num_points, 9), dtype=np.float32)
            centroid = np.zeros(3, dtype=np.float32)
        else:
            z = depth[vs, us]
            
            # Filter invalid depth (0.0 is invalid, >3.0 is likely background/noise)
            valid = (z > 0.001) & (z < 3.0)
            z = z[valid]
            vs = vs[valid]
            us = us[valid]

            x = (us - K[0,2]) * z / K[0,0]
            y = (vs - K[1,2]) * z / K[1,1]
            pts = np.stack([x, y, z], axis=-1)

            colors = rgb[vs, us]

            # Downsample via Open3D
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            pcd = pcd.voxel_down_sample(0.005)
            pcd.estimate_normals()

            normals = np.asarray(pcd.normals)
            pts = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            
            if len(pts) == 0:
                 # Fallback if voxel downsample killed everything
                 input_data = np.zeros((self.num_points, 9), dtype=np.float32)
                 centroid = np.zeros(3, dtype=np.float32)
            else:
                # Sample to fixed size
                N = len(pts)
                if N >= self.num_points:
                    idxs = np.random.choice(N, self.num_points, replace=False)
                else:
                    idxs = np.random.choice(N, self.num_points, replace=True)
                pts = pts[idxs]
                normals = normals[idxs]
                colors = colors[idxs]

                # Center the points
                centroid = np.mean(pts, axis=0)
                pts = pts - centroid

                # Augmentation: Random Jitter (Train only)
                if self.split_name == 'train':
                    jitter = np.random.normal(0, 0.002, pts.shape) # 2mm noise
                    pts = pts + jitter

                # The model input = XYZRGBNormal (9D)
                input_data = np.concatenate([pts, colors, normals], axis=1)

        # Compute GT pose
        T_wc = np.array(meta['extrinsic']).reshape(4,4)
        T_ow = np.array(meta['poses_world'][obj_id]).reshape(4,4)
        T_co = T_wc @ T_ow

        gt_R = T_co[:3,:3]
        gt_t = T_co[:3,3]
        
        # Residual translation
        gt_t_residual = gt_t - centroid

        # Extract object name
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

        scale = meta['scales'][obj_id]
        if scale is None:
            scale = np.array([1,1,1], dtype=np.float32)
        else:
            scale = np.array(scale, dtype=np.float32)
            if scale.ndim == 0:
                scale = np.array([scale, scale, scale], dtype=np.float32)

        return {
            'points': torch.from_numpy(input_data).float(),    # (N, 9)
            'gt_rot': torch.from_numpy(gt_R).float(),          # (3,3)
            'gt_t_residual': torch.from_numpy(gt_t_residual).float(),   # (3,)
            'centroid': torch.from_numpy(centroid).float(),
            'obj_id': obj_id,
            'sym_str': sym,
            'rgb_path': self.rgb_files[scene_idx],
            'intrinsic': K,
            'obj_dims': dims,
            'scale': torch.from_numpy(scale),
            'dataset_idx': idx
        }

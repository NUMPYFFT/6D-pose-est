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
    def __init__(self, split_name, data_dir, split_dir, num_points=4096, subset_size=None):
        self.split_name = split_name
        self.rgb_files, self.depth_files, self.label_files, self.meta_files = \
            get_split_files(split_name, data_dir, split_dir)
        self.num_points = num_points

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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scene_idx, obj_id = self.samples[idx]
        
        if self.use_preprocessed:
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
            # 1. Jitter
            pts = pts + np.random.normal(0, 0.002, pts.shape)
            
            # 2. Random Rotation (Perturbation +/- 30 deg)
            ang_x = np.random.uniform() * np.pi/3 - np.pi/6
            ang_y = np.random.uniform() * np.pi/3 - np.pi/6
            ang_z = np.random.uniform() * np.pi/3 - np.pi/6
            
            Rx = np.array([[1, 0, 0], [0, np.cos(ang_x), -np.sin(ang_x)], [0, np.sin(ang_x), np.cos(ang_x)]])
            Ry = np.array([[np.cos(ang_y), 0, np.sin(ang_y)], [0, 1, 0], [-np.sin(ang_y), 0, np.cos(ang_y)]])
            Rz = np.array([[np.cos(ang_z), -np.sin(ang_z), 0], [np.sin(ang_z), np.cos(ang_z), 0], [0, 0, 1]])
            
            R_aug = Rz @ Ry @ Rx
            
            pts = pts @ R_aug.T
            normals = normals @ R_aug.T
            gt_R = R_aug @ gt_R
            gt_t_residual = R_aug @ gt_t_residual

        # Final 9D input
        input_data = np.concatenate([pts, colors, normals], axis=1)

        return {
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

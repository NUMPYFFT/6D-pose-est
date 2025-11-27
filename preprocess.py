import os
import numpy as np
import torch
from PIL import Image
import pickle
import open3d as o3d
from tqdm import tqdm
import argparse
import config
from data import PoseDataset

def preprocess_dataset(args):
    split = args.split
    print(f"Preprocessing split: {split}")
    
    # Force use_preprocessed=False to load raw data
    dataset = PoseDataset(split, args.training_data_dir, args.split_dir, num_points=args.num_points)
    dataset.use_preprocessed = False # FORCE RAW LOADING
    
    # Create output directory
    output_dir = os.path.join(args.training_data_dir, "preprocessed", split)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving to: {output_dir}")
    
    for idx in tqdm(range(len(dataset))):
        scene_idx, obj_id = dataset.samples[idx]
        
        # Load images
        depth = np.asarray(Image.open(dataset.depth_files[scene_idx]), dtype=np.float32) / 1000.0
        rgb = np.asarray(Image.open(dataset.rgb_files[scene_idx]), dtype=np.float32) / 255.0
        label = np.asarray(Image.open(dataset.label_files[scene_idx]), dtype=np.int32)

        # Load metadata
        with open(dataset.meta_files[scene_idx], "rb") as f:
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

        # Extract depth
        z = depth[vs, us]

        # Filter invalid depth
        valid = (z > 0.001) & (z < 3.0)
        z = z[valid]
        vs = vs[valid]
        us = us[valid]

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
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=30))
            pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))

            pts = np.asarray(pcd.points).astype(np.float32)
            colors = np.asarray(pcd.colors).astype(np.float32)
            normals = np.asarray(pcd.normals).astype(np.float32)

        # Center
        centroid = np.mean(pts, axis=0).astype(np.float32)
        pts = pts - centroid

        # Ground truth pose
        T_wc = np.array(meta['extrinsic']).reshape(4,4)
        
        if split == 'test':
            gt_R = np.eye(3, dtype=np.float32)
            gt_t_residual = np.zeros(3, dtype=np.float32)
        else:
            T_ow = np.array(meta['poses_world'][obj_id]).reshape(4,4)
            T_co = T_wc @ T_ow

            gt_R = T_co[:3,:3].astype(np.float32)
            gt_t = T_co[:3,3]
            gt_t_residual = (gt_t - centroid).astype(np.float32)
        
        # Object info
        if obj_id in meta['object_ids']:
            idx2 = list(meta['object_ids']).index(obj_id)
            name = meta['object_names'][idx2]
        else:
            name = "unknown"

        if name in dataset.obj_info_cache:
            info = dataset.obj_info_cache[name]
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

        rgb_path = dataset.rgb_files[scene_idx]

        # Save to NPZ
        save_path = os.path.join(output_dir, f"{idx:06d}.npz")
        np.savez_compressed(save_path, 
                            points=pts, 
                            colors=colors, 
                            normals=normals, 
                            gt_R=gt_R, 
                            gt_t_residual=gt_t_residual, 
                            centroid=centroid, 
                            obj_id=obj_id, 
                            sym=sym, 
                            dims=dims, 
                            scale=scale, 
                            K=K, 
                            rgb_path=rgb_path)

if __name__ == "__main__":
    args = config.get_config()
    preprocess_dataset(args)
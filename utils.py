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
from data import PoseDataset

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
    corners_2d = corners_2d_hom[:, :2] / corners_2d_hom[:, 2:3]
    corners_2d = corners_2d.astype(int)
    
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
        if (corners_cam[edge[0], 2] > 0 and corners_cam[edge[1], 2] > 0 and
            -w_img < pt1[0] < 2*w_img and -h_img < pt1[1] < 2*h_img and
            -w_img < pt2[0] < 2*w_img and -h_img < pt2[1] < 2*h_img):
            cv2.line(image, pt1, pt2, color, thickness)

def preprocess_dataset(split, args):
    print(f"Processing split: {split}")
    # Initialize dataset to get the valid index list
    # We use num_points=4096 to ensure we get enough points
    dataset = PoseDataset(split, args.training_data_dir, args.split_dir, num_points=4096)
    
    save_dir = os.path.join(args.training_data_dir, "preprocessed", split)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Saving to {save_dir}...")
    
    for i, (scene_idx, obj_id) in enumerate(tqdm(dataset.samples)):
        # Re-implementing the loading logic to avoid augmentation and ensure consistency
        
        # Load images
        depth = np.asarray(Image.open(dataset.depth_files[scene_idx]), dtype=np.float32) / 1000.0
        rgb = np.asarray(Image.open(dataset.rgb_files[scene_idx]), dtype=np.float32) / 255.0
        label = np.asarray(Image.open(dataset.label_files[scene_idx]), dtype=np.int32)
        
        with open(dataset.meta_files[scene_idx], "rb") as f:
            meta = pickle.load(f)
            
        intr = meta['intrinsic']
        if isinstance(intr, dict):
            K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1]], dtype=np.float32)
        else:
            K = np.array(intr, dtype=np.float32)
            
        mask = (label == obj_id)
        vs, us = np.where(mask)
        
        z = depth[vs, us]
        valid = (z > 0.001) & (z < 3.0)
        z = z[valid]
        vs = vs[valid]
        us = us[valid]
        
        if len(z) == 0:
            continue # Should not happen due to indexing
            
        x = (us - K[0,2]) * z / K[0,0]
        y = (vs - K[1,2]) * z / K[1,1]
        pts = np.stack([x, y, z], axis=-1)
        colors = rgb[vs, us]
        
        # Downsample & Normals (The expensive part)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd = pcd.voxel_down_sample(0.005)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=30))
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
        
        pts = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        normals = np.asarray(pcd.normals)
        
        # Sample to 4096 (or less if not enough)
        # We save UP TO 4096 points.
        N = len(pts)
        if N >= 4096:
            idxs = np.random.choice(N, 4096, replace=False)
        else:
            idxs = np.random.choice(N, 4096, replace=True)
            
        pts = pts[idxs]
        colors = colors[idxs]
        normals = normals[idxs]
        
        # Centroid
        centroid = np.mean(pts, axis=0)
        pts = pts - centroid # Center the points
        
        # GT Pose
        T_wc = np.array(meta['extrinsic']).reshape(4,4)
        T_ow = np.array(meta['poses_world'][obj_id]).reshape(4,4)
        T_co = T_wc @ T_ow
        
        gt_R = T_co[:3,:3]
        gt_t = T_co[:3,3]
        gt_t_residual = gt_t - centroid
        
        # Object Info
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
        if scale is None: scale = np.array([1,1,1], dtype=np.float32)
        else: scale = np.array(scale, dtype=np.float32)
        if scale.ndim == 0: scale = np.array([scale, scale, scale], dtype=np.float32)

        # Save
        np.savez_compressed(
            os.path.join(save_dir, f"{i:06d}.npz"),
            points=pts.astype(np.float32),
            colors=colors.astype(np.float32),
            normals=normals.astype(np.float32),
            gt_R=gt_R.astype(np.float32),
            gt_t_residual=gt_t_residual.astype(np.float32),
            centroid=centroid.astype(np.float32),
            obj_id=obj_id,
            sym=str(sym),
            dims=dims,
            scale=scale,
            K=K,
            rgb_path=dataset.rgb_files[scene_idx] # For visualization if needed
        )

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


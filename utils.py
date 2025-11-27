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

if __name__ == "__main__":
    import config
    args = config.get_config()
    check_class_distribution(args)


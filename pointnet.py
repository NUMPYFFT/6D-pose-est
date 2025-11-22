# %%
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pickle
import open3d as o3d
from tqdm import tqdm
import pandas as pd
import itertools
import cv2
import utils

# %%
# --- Configuration ---
NUM_POINTS = 4096
BATCH_SIZE = 64 # Reduced batch size to fit larger point clouds in memory if needed, or keep 128 if 24GB is enough. 
# User said 24GB, 128 * 4096 * 9 * 4 bytes is small. 
# Let's keep 128 but maybe safe to go 64 if we add RGB layers. 
# Actually 24GB is huge. 128 is fine.
BATCH_SIZE = 96 
LEARNING_RATE = 0.001
EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {DEVICE}")

# %%
# --- Symmetry Utils (Adapted from benchmark_utils/pose_utils.py) ---
def get_rotation_matrix(axis, angle):
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    cos_angle, sin_angle = np.cos(angle), np.sin(angle)
    cross_prod_mat = np.cross(np.eye(3), axis)
    R = cos_angle * np.eye(3) + sin_angle * cross_prod_mat + (1.0 - cos_angle) * np.outer(axis, axis)
    return R

def get_symmetry_rotations(sym_axes, sym_orders):
    sym_rots_per_axis = []
    rot_axis = None
    for sym_axis, sym_order in zip(sym_axes, sym_orders):
        if sym_order is None:
            sym_rots_per_axis.append([np.eye(3)])
        elif np.isinf(sym_order):
            if rot_axis is None:
                rot_axis = sym_axis
            # Discretize continuous symmetry (e.g. 36 steps = 10 degrees)
            Rs = []
            for i in range(36):
                angle = i * (2 * np.pi / 36)
                R = get_rotation_matrix(sym_axis, angle)
                Rs.append(R)
            sym_rots_per_axis.append(Rs)
        else:
            Rs = []
            for i in range(sym_order):
                angle = i * (2 * np.pi / sym_order)
                R = get_rotation_matrix(sym_axis, angle)
                Rs.append(R)
            sym_rots_per_axis.append(Rs)
            
    sym_rots = []
    for Rs in itertools.product(*sym_rots_per_axis):
        R_tmp = np.eye(3)
        for R in Rs:
            R_tmp = R_tmp @ R
        sym_rots.append(R_tmp)
    return np.array(sym_rots), rot_axis

def parse_geometric_symmetry(sym_str):
    sym_axes = []
    sym_orders = []
    if pd.isna(sym_str) or str(sym_str).lower() == 'no' or str(sym_str).lower() == 'nan':
        return [np.array([0, 0, 0])], [None]
    
    parts = str(sym_str).split('|')
    for part in parts:
        part = part.strip().lower()
        if not part: continue
        
        if 'inf' in part:
            order = np.inf
            axis_char = part[0]
        else:
            try:
                digits = "".join([c for c in part if c.isdigit()])
                order = int(digits) if digits else 1
                axis_char = part[0]
            except:
                continue
        
        if axis_char == 'x':
            axis = np.array([1.0, 0.0, 0.0])
        elif axis_char == 'y':
            axis = np.array([0.0, 1.0, 0.0])
        elif axis_char == 'z':
            axis = np.array([0.0, 0.0, 1.0])
        else:
            continue
            
        sym_axes.append(axis)
        sym_orders.append(order)
        
    if not sym_axes:
        return [np.array([0, 0, 0])], [None]
        
    return sym_axes, sym_orders

def rotation_6d_to_matrix(rot_6d):
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks".
    Args:
        rot_6d: (B, 6) tensor
    Returns:
        rot_mat: (B, 3, 3) tensor
    """
    a1 = rot_6d[:, :3]
    a2 = rot_6d[:, 3:]
    
    b1 = F.normalize(a1, dim=1)
    b2 = a2 - torch.sum(b1 * a2, dim=1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    
    rot_mat = torch.stack([b1, b2, b3], dim=2) # (B, 3, 3)
    return rot_mat

# %%
# --- Data Loading Utils (Adapted from icp.py) ---
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

def depth_to_point_cloud(depth, intrinsic):
    z = depth
    v, u = np.indices(z.shape)
    uv1 = np.stack([u + 0.5, v + 0.5, np.ones_like(z)], axis=-1)
    points_viewer = uv1 @ np.linalg.inv(intrinsic).T * z[..., None]  # [H, W, 3]
    return points_viewer

# %%
# --- Dataset ---
class PoseDataset(Dataset):
    def __init__(self, split_name, num_points=1024, subset_size=300):
        self.split_name = split_name
        self.rgb_files, self.depth_files, self.label_files, self.meta_files = get_split_files(split_name)
        self.num_points = num_points
        self.objects_df = pd.read_csv("benchmark_utils/objects_v1.csv")
        
        # Cache object info for fast lookup
        self.obj_info_cache = {}
        for _, row in self.objects_df.iterrows():
            obj_name = str(row['object']).strip()
            sym = row['geometric_symmetry']
            
            # Manual Override for g_lego_duplo (ID 29)
            # It is a rectangular brick, so 180 rotation looks identical geometrically.
            if obj_name == 'g_lego_duplo':
                sym = 'z2|x2|y2'
            
            # Manual Override for e_lego_duplo (ID 20)
            if obj_name == 'e_lego_duplo':
                sym = 'z2|x2|y2'

            # Manual Override for nine_hole_peg_test (ID 43)
            if obj_name == 'nine_hole_peg_test':
                sym = 'z2|x2|y2'

            # Manual Override for mustard_bottle (ID 42)
            if obj_name == 'mustard_bottle':
                sym = 'z2'

            # Manual Override for bleach_cleanser (ID 8)
            if obj_name == 'bleach_cleanser':
                sym = 'z2'

            # Manual Override for c_toy_airplane (ID 13)
            # CSV says symmetric_z180, which is z2.
            # if obj_name == 'c_toy_airplane':
            #    sym = 'zinf|x2'
                
            # Manual Override for extra_large_clamp (ID 22)
            if obj_name == 'extra_large_clamp':
                sym = 'x2|z2'
                
            self.obj_info_cache[obj_name] = {
                'geometric_symmetry': sym,
                'width': row['width'],
                'length': row['length'],
                'height': row['height']
            }
        
        if subset_size is not None and subset_size < len(self.rgb_files):
            indices = np.random.choice(len(self.rgb_files), subset_size, replace=False)
            self.rgb_files = [self.rgb_files[i] for i in indices]
            self.depth_files = [self.depth_files[i] for i in indices]
            self.label_files = [self.label_files[i] for i in indices]
            self.meta_files = [self.meta_files[i] for i in indices]
            
        # Pre-calculate all valid (scene_idx, obj_id) pairs
        self.samples = []
        print("Indexing dataset...")
        for i in tqdm(range(len(self.meta_files)), desc="Loading metadata"):
            with open(self.meta_files[i], "rb") as f:
                meta = pickle.load(f)
            
            obj_ids = meta['object_ids']
            valid_objs = [oid for oid in obj_ids if meta['poses_world'][oid] is not None]
            
            for oid in valid_objs:
                self.samples.append((i, oid))
        print(f"Indexed {len(self.samples)} samples from {len(self.meta_files)} scenes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scene_idx, target_obj_id = self.samples[idx]
        
        # Load Data
        depth = np.asarray(Image.open(self.depth_files[scene_idx]), dtype=np.float32) / 1000.0
        rgb = np.asarray(Image.open(self.rgb_files[scene_idx]), dtype=np.float32) / 255.0 # Load RGB
        label = np.asarray(Image.open(self.label_files[scene_idx]), dtype=np.int32)
        with open(self.meta_files[scene_idx], "rb") as f:
            meta = pickle.load(f)

        intrinsic = meta['intrinsic']
        if isinstance(intrinsic, dict):
            K = np.array([[intrinsic['fx'], 0, intrinsic['cx']], [0, intrinsic['fy'], intrinsic['cy']], [0, 0, 1]])
        else:
            K = np.array(intrinsic)

        # Optimization: Only project masked points to 3D
        mask = (label == target_obj_id)
        vs, us = np.where(mask)
        
        # If object not found or too small, pick another sample
        if len(vs) < 50:
             return self.__getitem__(np.random.randint(0, len(self)))
             
        zs = depth[vs, us]
        colors = rgb[vs, us] # Sample RGB colors
        
        # Project to 3D
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        xs = (us + 0.5 - cx) * zs / fx
        ys = (vs + 0.5 - cy) * zs / fy
        obj_points = np.stack([xs, ys, zs], axis=-1)

        # Sampling (Borrowing Voxel Downsampling from ICP)
        # Use Open3D for voxel downsampling to get uniform points
        obj_normals = np.zeros((0, 3), dtype=np.float32)
        
        if len(obj_points) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(obj_points)
            pcd.colors = o3d.utility.Vector3dVector(colors) # Pass colors to PCD
            
            pcd = pcd.voxel_down_sample(voxel_size=0.005) # 5mm voxel size
            
            # Estimate Normals
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
            pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
            
            obj_points = np.asarray(pcd.points)
            obj_normals = np.asarray(pcd.normals)
            obj_colors = np.asarray(pcd.colors) # Retrieve downsampled colors

        # Simple sampling strategy
        if len(obj_points) == 0:
            # Should not happen if we filtered correctly, but just in case
            points = np.zeros((self.num_points, 3), dtype=np.float32)
            normals = np.zeros((self.num_points, 3), dtype=np.float32)
            colors_sampled = np.zeros((self.num_points, 3), dtype=np.float32)
            normals[:, 2] = 1.0 # Default normal z-up
        elif len(obj_points) >= self.num_points:
            choice = np.random.choice(len(obj_points), self.num_points, replace=False)
            points = obj_points[choice]
            normals = obj_normals[choice]
            colors_sampled = obj_colors[choice]
        else:
            choice = np.random.choice(len(obj_points), self.num_points, replace=True)
            points = obj_points[choice]
            normals = obj_normals[choice]
            colors_sampled = obj_colors[choice]

        # Normalize Points (Center them)
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        
        # Augmentation: Jitter (Add noise)
        if self.split_name == "train":
            noise = np.random.normal(0, 0.002, points_centered.shape) # 2mm noise
            points_centered += noise

        # Stack Points and Normals (N, 6)
        input_data = np.concatenate([points_centered, normals], axis=1)

        # Get GT Pose
        T_wc = np.array(meta['extrinsic']).reshape(4, 4)
        T_ow = np.array(meta['poses_world'][target_obj_id]).reshape(4, 4)
        T_co = T_wc @ T_ow # Object to Camera
        
        gt_rotation = T_co[:3, :3]
        gt_translation = T_co[:3, 3]

        # For translation regression, we predict the residual from the centroid
        t_residual = gt_translation - centroid

        # Get Object Info (Name, Symmetry, Dimensions)
        # Find object name from ID
        if target_obj_id in meta['object_ids']:
            obj_idx_in_list = list(meta['object_ids']).index(target_obj_id)
            obj_name = meta['object_names'][obj_idx_in_list]
        else:
            obj_name = "unknown" 

        # Lookup in DF
        if obj_name != "unknown" and obj_name in self.obj_info_cache:
            info = self.obj_info_cache[obj_name]
            sym_str = info['geometric_symmetry']
            dims = np.array([info['width'], info['length'], info['height']], dtype=np.float32)
        else:
            sym_str = "no"
            dims = np.zeros(3, dtype=np.float32)
        
        # Handle Scale
        scale_data = meta['scales'][target_obj_id]
        if scale_data is None:
            scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        else:
            scale = np.array(scale_data, dtype=np.float32)
            if scale.ndim == 0:
                scale = np.array([scale, scale, scale], dtype=np.float32)
            elif scale.size == 1:
                s = scale.item()
                scale = np.array([s, s, s], dtype=np.float32)
            elif scale.size == 3:
                pass # Already (3,)
            else:
                scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        return {
            'points': torch.from_numpy(input_data).float(), # (N, 6)
            'centroid': torch.from_numpy(centroid).float(),
            'gt_rot': torch.from_numpy(gt_rotation).float(), # (3, 3)
            'gt_t_residual': torch.from_numpy(t_residual).float(),
            'obj_id': target_obj_id,
            'sym_str': str(sym_str),
            'rgb_path': self.rgb_files[scene_idx],
            'intrinsic': K.astype(np.float32),
            'obj_dims': dims,
            'scale': torch.from_numpy(scale),
            'dataset_idx': idx
        }

# %%
# --- Model ---
class PointNet(nn.Module):
    def __init__(self, num_classes=79):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(6, 64, 1) # Input channels: 6 (XYZ + Normals)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # Embedding for Object ID
        self.obj_emb = nn.Embedding(num_classes + 1, 128) # +1 for safety

        # Decoupled Heads with Dropout
        self.rot_head = nn.Sequential(
            nn.Linear(1024 + 128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 6)
        )

        self.trans_head = nn.Sequential(
            nn.Linear(1024 + 128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)
        )

    def forward(self, x, obj_id):
        # x: (B, N, 6) -> (B, 6, N)
        x = x.transpose(2, 1)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # Object Embedding
        emb = self.obj_emb(obj_id) # (B, 128)
        
        # Concatenate
        x = torch.cat([x, emb], dim=1) # (B, 1152)

        rot = self.rot_head(x)
        trans = self.trans_head(x)
        
        return rot, trans

# %%
# %%
def visualize_scenes(model, dataset, epoch, device, scene_indices=None, highlight_targets=None, num_scenes=5):
    model.eval()
    
    if scene_indices is None:
        # Get all unique scene indices present in dataset.samples
        unique_scene_indices = list(set([s[0] for s in dataset.samples]))
        if not unique_scene_indices: return
        
        # Pick random scenes
        chosen_indices = np.random.choice(unique_scene_indices, min(num_scenes, len(unique_scene_indices)), replace=False)
    else:
        chosen_indices = scene_indices
    
    with torch.no_grad():
        for scene_idx in chosen_indices:
            # Find all sample indices for this scene
            sample_indices = [i for i, s in enumerate(dataset.samples) if s[0] == scene_idx]
            
            # Load Image
            rgb_path = dataset.rgb_files[scene_idx]
            img = cv2.imread(rgb_path)
            if img is None: continue
            
            for idx in sample_indices:
                data = dataset[idx]
                
                # Check if we got the right scene (in case of fallback in __getitem__)
                if data['rgb_path'] != rgb_path:
                    continue
                    
                points = data['points'].unsqueeze(0).to(device)
                obj_id = torch.tensor([data['obj_id']]).to(device)
                
                pred_rot, pred_trans = model(points, obj_id)
                
                p_rot_6d = pred_rot[0].unsqueeze(0)
                p_R = rotation_6d_to_matrix(p_rot_6d)[0].cpu().numpy()
                
                p_t_res = pred_trans[0].cpu().numpy()
                centroid = data['centroid'].numpy()
                p_t = centroid + p_t_res
                
                dims = data['obj_dims'] # Already numpy
                scale = data['scale'].numpy()
                size = dims * scale
                K = data['intrinsic']
                
                # Determine color
                # Default Green
                color = (0, 255, 0) 
                
                # If this object is a target for highlighting (high error), make it Red
                if highlight_targets and scene_idx in highlight_targets:
                    if data['obj_id'] in highlight_targets[scene_idx]:
                        color = (0, 0, 255) # Red (BGR)
                
                # Draw GT Box (Blue)
                gt_R = data['gt_rot'].numpy()
                gt_t_res = data['gt_t_residual'].numpy()
                gt_t = centroid + gt_t_res
                utils.draw_projected_box3d(img, gt_t, size, gt_R, np.eye(4), K, color=(255, 0, 0), thickness=1)
                
                # Draw Pred Box
                utils.draw_projected_box3d(img, p_t, size, p_R, np.eye(4), K, color=color, thickness=2)
                
                # Draw Pred Axes
                utils.draw_axes(img, p_t, p_R, np.eye(4), K, length=max(dims)/2, thickness=2)
                
            cv2.imwrite(f"training_vis/epoch_{epoch+1}_scene_{scene_idx}.png", img)
            
    model.train()

# %%
# --- Loss Functions ---
def chamfer_distance(p1, p2):
    """
    Compute Chamfer Distance between two point clouds.
    p1: (B, N, 3)
    p2: (B, M, 3)
    Returns: (B,)
    """
    # p1 = p1.unsqueeze(2) # (B, N, 1, 3)
    # p2 = p2.unsqueeze(1) # (B, 1, M, 3)
    # dist = torch.norm(p1 - p2, dim=-1) # (B, N, M) - OOM for large N
    
    # Efficient implementation using batching or smaller chunks if needed
    # For N=1024, B=32, N*N*B*4 bytes = 128MB. It fits.
    
    dists = torch.cdist(p1, p2) # (B, N, M)
    
    min_dist1, _ = torch.min(dists, dim=2) # (B, N) - dist from p1 to closest in p2
    min_dist2, _ = torch.min(dists, dim=1) # (B, M) - dist from p2 to closest in p1
    
    return torch.mean(min_dist1, dim=1) + torch.mean(min_dist2, dim=1)

def point_matching_loss(pred_R, pred_t, gt_R, gt_t, points, sym_strs):
    """
    pred_R: (B, 3, 3)
    pred_t: (B, 3)
    gt_R: (B, 3, 3)
    gt_t: (B, 3)
    points: (B, N, 3) - Centered points
    sym_strs: list of strings
    """
    # Transform points
    # (B, N, 3) @ (B, 3, 3).T -> (B, N, 3)
    pred_pts = torch.bmm(points, pred_R.transpose(1, 2)) + pred_t.unsqueeze(1)
    gt_pts = torch.bmm(points, gt_R.transpose(1, 2)) + gt_t.unsqueeze(1)
    
    losses = []
    
    # We can process in batch if all are same type, but here we have mixed.
    # Optimization: Group by symmetry type?
    # Or just iterate.
    
    # For efficiency, let's compute L2 for all, and overwrite for symmetric.
    # L2 (ADD)
    l2_dists = torch.norm(pred_pts - gt_pts, dim=2).mean(dim=1) # (B,)
    
    # Check for symmetric objects
    # If any object is symmetric, we must calculate Chamfer for it.
    # To avoid loop, we can compute Chamfer for ALL, but that's slow.
    # Let's iterate and pick.
    
    final_losses = []
    for i in range(len(sym_strs)):
        s = sym_strs[i]
        if 'inf' in s or '2' in s or '4' in s: # Symmetric
            # Use Chamfer
            # Subsample for speed if N is large
            # Using 512 points for loss calculation
            idx = torch.randperm(points.shape[1])[:512]
            p_sub = pred_pts[i:i+1, idx, :]
            g_sub = gt_pts[i:i+1, idx, :]
            loss = chamfer_distance(p_sub, g_sub)
            final_losses.append(loss)
        else:
            # Use ADD (already computed)
            final_losses.append(l2_dists[i:i+1])
            
    return torch.cat(final_losses).mean()

# --- Training ---
def train():
    # Use subset of dataset for faster training
    train_dataset = PoseDataset("train", num_points=NUM_POINTS, subset_size=None)
    # Increased num_workers for GPU utilization
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, persistent_workers=False)
    
    model = PointNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Loss functions
    # L1 Loss for translation
    trans_criterion = nn.L1Loss()
    
    # Create output dir for visualization
    os.makedirs("training_vis", exist_ok=True)
    
    print("Starting training...")
    model.train()
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        total_loss = 0
        total_rot_err = 0.0
        total_trans_err = 0.0
        epoch_errors = []
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            points = batch['points'].to(DEVICE) # (B, N, 9)
            gt_rot = batch['gt_rot'].to(DEVICE) # (B, 3, 3)
            gt_t_residual = batch['gt_t_residual'].to(DEVICE)
            obj_ids = batch['obj_id'].to(DEVICE)
            sym_strs = batch['sym_str']
            dataset_indices = batch['dataset_idx']
            
            # Extract XYZ for loss calculation
            points_xyz = points[:, :, :3] # (B, N, 3)
            
            optimizer.zero_grad()
            
            pred_rot, pred_trans = model(points, obj_ids)
            
            # Convert Pred 6D to Matrix
            pred_R = rotation_6d_to_matrix(pred_rot) # (B, 3, 3)
            
            # Point Matching Loss (Handles Symmetry)
            loss_pm = point_matching_loss(pred_R, pred_trans, gt_rot, gt_t_residual, points_xyz, sym_strs)
            
            # Regularization for translation (optional, but PM loss covers it)
            # We can add a small direct translation loss to help convergence
            loss_t = trans_criterion(pred_trans, gt_t_residual)
            
            # Total Loss
            # PM Loss is in meters. 
            # 1cm error = 0.01. 
            # We want gradients to be significant. Scale up.
            loss = 100.0 * loss_pm + 1.0 * loss_t
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate Metrics for Logging
            with torch.no_grad():
                # Compute ADD/ADD-S per item for logging
                # This gives a better sense of quality than raw rotation error for symmetric objects
                
                # Transform points
                pred_pts = torch.bmm(points_xyz, pred_R.transpose(1, 2)) + pred_trans.unsqueeze(1)
                gt_pts = torch.bmm(points_xyz, gt_rot.transpose(1, 2)) + gt_t_residual.unsqueeze(1)
                
                batch_add_errs = []
                
                for i in range(len(points)):
                    s = sym_strs[i]
                    if 'inf' in s or '2' in s or '4' in s: # Symmetric
                        # ADD-S (Chamfer)
                        # Use subset for speed
                        idx = torch.randperm(points.shape[1])[:512]
                        p_sub = pred_pts[i:i+1, idx, :]
                        g_sub = gt_pts[i:i+1, idx, :]
                        # Chamfer
                        dists = torch.cdist(p_sub, g_sub)
                        min_dist1, _ = torch.min(dists, dim=2)
                        min_dist2, _ = torch.min(dists, dim=1)
                        err = (torch.mean(min_dist1) + torch.mean(min_dist2)).item() / 2.0 # Average distance
                    else:
                        # ADD
                        err = torch.norm(pred_pts[i] - gt_pts[i], dim=1).mean().item()
                    
                    batch_add_errs.append(err)
                    epoch_errors.append((err, dataset_indices[i].item()))
                
                batch_avg_add = sum(batch_add_errs) / len(batch_add_errs)
                total_rot_err += batch_avg_add # Using this variable for ADD error now
            
            total_trans_err += torch.norm(pred_trans - gt_t_residual, dim=1).mean().item()
            
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        avg_loss = total_loss / len(train_loader)
        avg_add_err = total_rot_err / len(train_loader)
        avg_trans_err = total_trans_err / len(train_loader)
        
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}, Avg ADD/ADD-S Err: {avg_add_err:.4f} m, Avg Trans Err: {avg_trans_err:.4f} m, LR: {current_lr:.6f}")
        
        # Identify worst scenes (by ADD Error)
        epoch_errors.sort(key=lambda x: x[0], reverse=True)
        top_k_errors = epoch_errors[:5]
        
        print("Top 5 Worst ADD/ADD-S Errors:")
        worst_scene_indices = []
        highlight_targets = {} 
        
        for err, idx in top_k_errors:
            scene_idx, obj_id = train_dataset.samples[idx]
            print(f"  Error: {err:.4f} m | Scene: {scene_idx} | Obj ID: {obj_id}")
            
            if scene_idx not in worst_scene_indices:
                worst_scene_indices.append(scene_idx)
            
            if scene_idx not in highlight_targets:
                highlight_targets[scene_idx] = []
            highlight_targets[scene_idx].append(obj_id)

        # Visualize Scenes
        visualize_scenes(model, train_dataset, epoch, DEVICE, scene_indices=worst_scene_indices, highlight_targets=highlight_targets)
        
        # Save Best Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "pointnet_best.pth")
            print(f"New best model saved with loss: {best_loss:.4f}")

# %%
if __name__ == "__main__":
    train()

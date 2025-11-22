import numpy as np
import torch
import torch.nn.functional as F
import cv2
import pandas as pd
import torch.nn as nn
import itertools

# --- Rotation conversions ---
def rotation_6d_to_matrix(rot_6d):
    batch_size = rot_6d.shape[0]
    a1 = rot_6d[:, :3]
    a2 = rot_6d[:, 3:]
    b1 = nn.functional.normalize(a1, dim=1)
    b2 = a2 - torch.sum(b1 * a2, dim=1, keepdim=True) * b1
    b2 = nn.functional.normalize(b2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    rot_matrix = torch.stack([b1, b2, b3], dim=1).transpose(-2, -1)
    return rot_matrix

# --- Symmetry matrix generation ---
def get_symmetry_rotation_matrices(symmetry_type, device, num_samples=8):
    dtype = torch.float32
    if symmetry_type == 'symmetric_z':
        angles = torch.linspace(0, 2 * np.pi, num_samples, device=device, dtype=dtype)
        rotations = []
        for angle in angles:
            cos_a, sin_a = torch.cos(angle), torch.sin(angle)
            rotations.append(torch.tensor([
                [cos_a, -sin_a, 0.0],
                [sin_a, cos_a, 0.0],
                [0.0, 0.0, 1.0]
            ], device=device, dtype=dtype))
        return torch.stack(rotations)

    if symmetry_type == 'symmetric_z90':
        return torch.tensor([
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        ], device=device, dtype=dtype)

    if symmetry_type == 'symmetric_x180':
        return torch.tensor([
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
        ], device=device, dtype=dtype)

    return torch.eye(3, device=device).unsqueeze(0)




def translation_distance(pred_trans, gt_trans):
    """
    Compute Euclidean distance between translations.
    
    Args:
        pred_trans: [batch_size, 3] predicted translations
        gt_trans: [batch_size, 3] ground truth translations
        
    Returns:
        distance: [batch_size] distances in same units as input
    """
    return torch.norm(pred_trans - gt_trans, dim=1)


def compute_symmetry_aware_rotation_errors(pred_rotmat, gt_rotmat, object_names, symmetry_dict, device=None):
    """Compute minimal rotation error over object symmetries for each sample using rotation matrices."""
    if device is None:
        device = pred_rotmat.device

    pred_rotmat = pred_rotmat.to(device)
    gt_rotmat = gt_rotmat.to(device)

    errors = []
    batch_size = pred_rotmat.shape[0]
    object_names = object_names or []

    for i in range(batch_size):
        # --- Pose Metrics ---
        obj_name = object_names[i] if i < len(object_names) else None
        symmetry_type = symmetry_dict.get(obj_name, 'asymmetric') if symmetry_dict else 'asymmetric'

    gt_rot_matrix = gt_rotmat[i]  # [3, 3]
    symmetry_rots = get_symmetry_rotation_matrices(symmetry_type, device)  # [N, 3, 3]
    # Compute all symmetric ground truth rotations: [N, 3, 3]
    # Each symmetric rotation: symmetry_rot @ gt_rot_matrix
    symmetric_rotations = torch.matmul(symmetry_rots, gt_rot_matrix)  # [N, 3, 3]

    pred_matrix = pred_rotmat[i].unsqueeze(0).expand_as(symmetric_rotations)  # [N, 3, 3]
    # Compute rotation error for each symmetric equivalent
    angular_errors = rotation_matrix_angular_distance(pred_matrix, symmetric_rotations)
    errors.append(torch.min(angular_errors).item())

    return errors


def load_symmetry_info(csv_path='benchmark_utils/objects.csv'):
    """Load symmetry metadata mapping object name to symmetry type."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return {}

    return {row['object']: row['metric'] for _, row in df.iterrows()}


def rotation_matrix_angular_distance(pred_rotmat, gt_rotmat):
    """
    Compute angular distance between rotation matrices in degrees.
    Args:
        pred_rotmat: [batch_size, 3, 3] predicted rotation matrices
        gt_rotmat: [batch_size, 3, 3] ground truth rotation matrices
    Returns:
        angular_distance: [batch_size] angular distances in degrees
    """
    # Compute relative rotation matrix
    rel_rot = torch.matmul(pred_rotmat.transpose(1, 2), gt_rotmat)
    # Compute trace
    trace = rel_rot[:, 0, 0] + rel_rot[:, 1, 1] + rel_rot[:, 2, 2]
    # Clamp trace for numerical stability
    trace = torch.clamp((trace - 1) / 2, -1.0, 1.0)
    angular_distance_rad = torch.acos(trace)
    angular_distance_deg = angular_distance_rad * 180.0 / np.pi
    return angular_distance_deg

def pose_loss(pred_rotmat, pred_translation, gt_rotmat, gt_translation, 
              rotation_weight=1.0, translation_weight=1.0):
    """
    Combined pose loss function using rotation matrices.
    Args:
        pred_rotmat: [batch_size, 3, 3] predicted rotation matrices
        pred_translation: [batch_size, 3] predicted translations
        gt_rotmat: [batch_size, 3, 3] ground truth rotation matrices
        gt_translation: [batch_size, 3] ground truth translations
        rotation_weight: Weight for rotation loss
        translation_weight: Weight for translation loss
    Returns:
        loss: Combined pose loss
        metrics: Dictionary with individual losses and errors
    """
    # Rotation loss using angular distance
    rotation_loss = rotation_matrix_angular_distance(pred_rotmat, gt_rotmat).mean()
    # Translation loss using L2 distance
    translation_loss = F.mse_loss(pred_translation, gt_translation)
    # Combined loss
    total_loss = rotation_weight * rotation_loss + translation_weight * translation_loss
    # Compute metrics for monitoring
    with torch.no_grad():
        rotation_error = rotation_matrix_angular_distance(pred_rotmat, gt_rotmat)
        translation_error = translation_distance(pred_translation, gt_translation)
        metrics = {
            'rotation_loss': rotation_loss.item(),
            'translation_loss': translation_loss.item(),
            'rotation_error_deg': rotation_error.mean().item(),
            'translation_error_m': translation_error.mean().item(),
            'rotation_error_max': rotation_error.max().item(),
            'translation_error_max': translation_error.max().item()
        }
        # --- Point Cloud and Visualization Utilities ---
    return total_loss, metrics


def rotation_geodesic_loss(pred_R, gt_R):
    """
    pred_R, gt_R: (B, 3, 3)
    Returns angle error in radians: (B,)
    """
    # R_rel = R_pred^T * R_gt
    R_rel = torch.bmm(pred_R.transpose(1, 2), gt_R)

    # trace(R_rel)
    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]

    # Avoid numerical issues
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)

    # angle in radians
    theta = torch.acos(cos_theta)
    return theta


def compute_pose_metrics(pred_rotmat, pred_translation, gt_rotmat, gt_translation,
                         object_names=None, symmetry_dict=None):
    """
    Compute comprehensive pose estimation metrics using rotation matrices.
    Returns:
        metrics: Dictionary with rotation and translation errors
    """
    with torch.no_grad():
        if symmetry_dict and object_names is not None:
            rotation_error_list = compute_symmetry_aware_rotation_errors(
                pred_rotmat, gt_rotmat, object_names, symmetry_dict, pred_rotmat.device
            )
            rotation_errors = torch.tensor(rotation_error_list, device=pred_rotmat.device)
        else:
            rotation_errors = rotation_matrix_angular_distance(pred_rotmat, gt_rotmat)
        # Translation errors in meters (assuming input is in meters)
        translation_errors = translation_distance(pred_translation, gt_translation)
        metrics = {
            'rotation_mean': rotation_errors.mean().item(),
            'rotation_median': rotation_errors.median().item(),
            'rotation_std': rotation_errors.std().item() if rotation_errors.numel() > 1 else 0.0,
            'rotation_max': rotation_errors.max().item(),
            'translation_mean': translation_errors.mean().item(),
            'translation_median': translation_errors.median().item(),
            'translation_std': translation_errors.std().item() if translation_errors.numel() > 1 else 0.0,
            'translation_max': translation_errors.max().item(),
            # Success rates (professor's requirements)
            'rotation_success_4deg': (rotation_errors <= 4.0).float().mean().item(),
            'translation_success_1cm': (translation_errors <= 0.01).float().mean().item(),
            'combined_success': ((rotation_errors <= 4.0) & (translation_errors <= 0.01)).float().mean().item()
        }
    return metrics


def apply_pose_to_points(points, rotation_matrix, translation):
    """
    Apply pose transformation to point cloud using rotation matrices.
    Args:
        points: [batch_size, 3, num_points] point clouds
        rotation_matrix: [batch_size, 3, 3] rotation matrices
        translation: [batch_size, 3] translations
    Returns:
        transformed_points: [batch_size, 3, num_points] transformed point clouds
    """
    batch_size, _, num_points = points.shape
    # Apply rotation: R @ points
    rotated_points = torch.bmm(rotation_matrix, points)  # [batch_size, 3, num_points]
    # Apply translation
    translation_expanded = translation.unsqueeze(2).expand(-1, -1, num_points)
    transformed_points = rotated_points + translation_expanded
    return transformed_points


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


def project_3d_to_2d(points_3d, intrinsics):
    """
    Project 3D points to 2D image coordinates.
    
    Args:
        points_3d: [N, 3] 3D points in camera coordinates
        intrinsics: [3, 3] camera intrinsic matrix
        
    Returns:
        points_2d: [N, 2] 2D image coordinates
    """
    # Convert to homogeneous coordinates
    ones = np.ones((points_3d.shape[0], 1))
    points_3d_homogeneous = np.hstack([points_3d, ones])
    
    # Project to 2D
    points_2d_homogeneous = intrinsics @ points_3d.T
    
    # Convert from homogeneous to cartesian
    points_2d = points_2d_homogeneous[:2, :].T / points_2d_homogeneous[2, :].T[:, np.newaxis]
    
    return points_2d


def get_3d_bbox_corners(size, pose_rotation, pose_translation):
    """
    Get 3D bounding box corners from size and pose.
    Args:
        size: [3] box dimensions [width, height, depth]
        pose_rotation: [6] 6D rotation representation
        pose_translation: [3] translation vector
    Returns:
        corners: [8, 3] 3D box corners in world coordinates
    """
    # Define box corners in object coordinate system (centered at origin)
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
    # Convert 6D rotation to rotation matrix
    rotation_matrix = rotation_6d_to_matrix(torch.tensor(pose_rotation).unsqueeze(0)).squeeze(0).cpu().numpy()
    # Transform corners to world coordinates
    corners_world = (rotation_matrix @ corners_obj.T).T + pose_translation
    return corners_world

def draw_axes(image, center, rotation, extrinsic, intrinsic, length=0.1, thickness=2):
    """
    Draw 3D coordinate axes on an image.
    Args:
        image: Input image (numpy array) - modified in place
        center: [3] 3D center position in world coordinates
        rotation: [3, 3] rotation matrix (object to world)
        extrinsic: [4, 4] camera extrinsic matrix (world to camera)
        intrinsic: [3, 3] camera intrinsic matrix
        length: Length of the axes in meters
        thickness: Line thickness
    """
    # Define axes in object coordinates
    # X: Red, Y: Green, Z: Blue
    axes_obj = np.array([
        [0.0, 0.0, 0.0], # Origin
        [length, 0.0, 0.0], # X
        [0.0, length, 0.0], # Y
        [0.0, 0.0, length]  # Z
    ])
    
    # Transform to world coordinates
    axes_world = (rotation @ axes_obj.T).T + center
    
    # Add homogeneous coordinate
    axes_world_hom = np.hstack([axes_world, np.ones((4, 1))])
    
    # Transform to camera coordinates
    axes_cam_hom = (extrinsic @ axes_world_hom.T).T
    axes_cam = axes_cam_hom[:, :3]
    
    # Project to 2D
    axes_2d_hom = (intrinsic @ axes_cam.T).T
    axes_2d = axes_2d_hom[:, :2] / axes_2d_hom[:, 2:3]
    axes_2d = axes_2d.astype(int)
    
    origin = tuple(axes_2d[0])
    pt_x = tuple(axes_2d[1])
    pt_y = tuple(axes_2d[2])
    pt_z = tuple(axes_2d[3])
    
    h_img, w_img = image.shape[:2]
    
    def is_valid(pt):
        return -w_img < pt[0] < 2*w_img and -h_img < pt[1] < 2*h_img

    if axes_cam[0, 2] > 0: # Origin in front of camera
        if is_valid(origin) and is_valid(pt_x):
            cv2.line(image, origin, pt_x, (0, 0, 255), thickness) # X - Red (BGR)
        if is_valid(origin) and is_valid(pt_y):
            cv2.line(image, origin, pt_y, (0, 255, 0), thickness) # Y - Green
        if is_valid(origin) and is_valid(pt_z):
            cv2.line(image, origin, pt_z, (255, 0, 0), thickness) # Z - Blue

# --- Symmetry Utils ---
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

def compute_symmetry_aware_loss(R_pred, R_gt, sym_str):
    sym_axes, sym_orders = parse_geometric_symmetry(sym_str)
    sym_rots, rot_axis = get_symmetry_rotations(sym_axes, sym_orders)
    
    losses = []
    
    # If infinite symmetry (e.g. cylinder)
    if rot_axis is not None:
        for R_sym in sym_rots:
            R_gt_sym = R_gt @ R_sym
            # Axis alignment error
            axis_pred = R_pred @ rot_axis
            axis_gt = R_gt_sym @ rot_axis
            dot = np.clip(np.dot(axis_pred, axis_gt), -1.0, 1.0)
            angle = np.arccos(dot)
            losses.append(np.degrees(angle))
    else:
        # Discrete symmetry
        for R_sym in sym_rots:
            R_gt_sym = R_gt @ R_sym
            # Rotation error
            trace = np.trace(R_pred.T @ R_gt_sym)
            angle = np.arccos(np.clip(0.5 * (trace - 1), -1.0, 1.0))
            losses.append(np.degrees(angle))
            
    return min(losses) if losses else 0.0

# --- Loss Functions ---
def chamfer_distance(p1, p2):
    """
    Compute Chamfer Distance between two point clouds.
    p1: (B, N, 3)
    p2: (B, M, 3)
    Returns: (B,)
    """
    dists = torch.cdist(p1, p2) # (B, N, M)
    
    min_dist1, _ = torch.min(dists, dim=2) # (B, N) - dist from p1 to closest in p2
    min_dist2, _ = torch.min(dists, dim=1) # (B, M) - dist from p2 to closest in p1
    
    return torch.mean(min_dist1, dim=1) + torch.mean(min_dist2, dim=1)

def point_matching_loss(pred_R, pred_t, gt_R, gt_t, points, sym_strs):
    """
    pred_R: (B, 3, 3)
    pred_t: (B, 3)
    gt_R:   (B, 3, 3)
    gt_t:   (B, 3)
    points: (B, N, 3)
    sym_strs: list of strings
    """
    B, N, _ = points.shape

    # Transform predicted & gt point clouds
    pred_pts = torch.bmm(points, pred_R.transpose(1, 2)) + pred_t.unsqueeze(1)
    gt_pts   = torch.bmm(points, gt_R.transpose(1, 2))   + gt_t.unsqueeze(1)

    final_losses = []

    for i in range(B):
        s = sym_strs[i]

        # ----- SYMMETRIC OBJECT (ADD-S) -----
        if ('2' in s) or ('4' in s) or ('inf' in s):
            idx = torch.randperm(N)[:512]
            p_sub = pred_pts[i:i+1, idx, :]
            g_sub = gt_pts[i:i+1, idx, :]

            # Chamfer
            d = torch.cdist(p_sub, g_sub)       # (1,512,512)
            d1 = d.min(dim=2)[0].mean()
            d2 = d.min(dim=1)[0].mean()
            loss = 0.5 * (d1 + d2)

        # ----- NON-SYMMETRIC (ADD) -----
        else:
            loss = torch.norm(pred_pts[i] - gt_pts[i], dim=1).mean()

        # ensure shape = (1,)
        final_losses.append(loss.unsqueeze(0))

    return torch.cat(final_losses).mean()

def get_symmetry_matrices_torch(sym_str, device):
    """
    Generates a batch of symmetry rotation matrices from a symmetry string (e.g., 'z2|x2').
    Returns: (K, 3, 3) tensor
    """
    if not sym_str or sym_str == 'no' or sym_str == 'asymmetric' or 'nan' in sym_str:
        return torch.eye(3, device=device).unsqueeze(0)

    # Base set of rotations (Identity)
    rotations = [torch.eye(3, device=device)]

    parts = sym_str.split('|')
    for part in parts:
        part = part.strip().lower()
        if not part: continue

        # Parse Axis
        if part.startswith('x'): axis = torch.tensor([1., 0., 0.], device=device)
        elif part.startswith('y'): axis = torch.tensor([0., 1., 0.], device=device)
        elif part.startswith('z'): axis = torch.tensor([0., 0., 1.], device=device)
        else: continue

        # Parse Order
        if 'inf' in part:
            # For infinite symmetry, we cannot enumerate all. 
            # We return current set (this function is for discrete mainly).
            # Handling inf in loss requires separate logic (axis projection).
            continue
        else:
            try:
                digits = "".join([c for c in part if c.isdigit()])
                order = int(digits) if digits else 1
            except:
                continue

        # Generate new rotations for this axis
        new_rots = []
        for k in range(1, order):
            angle = 2 * np.pi * k / order
            
            # Rodrigues formula for rotation matrix
            K = torch.tensor([
                [0., -axis[2], axis[1]],
                [axis[2], 0., -axis[0]],
                [-axis[1], axis[0], 0.]
            ], device=device)
            
            # Fix: Ensure angle is on correct device for sin/cos
            angle_t = torch.tensor(angle, device=device)
            R = torch.eye(3, device=device) + torch.sin(angle_t) * K + (1 - torch.cos(angle_t)) * (K @ K)
            new_rots.append(R)

        # Combine with existing rotations (Cartesian product)
        current_rots = list(rotations)
        for r_new in new_rots:
            for r_curr in current_rots:
                rotations.append(r_curr @ r_new)

    return torch.stack(rotations)

def symmetry_aware_geodesic_loss(pred_R, gt_R, sym_strs):
    """
    Computes the minimum geodesic loss over all valid symmetric poses.
    pred_R: (B, 3, 3)
    gt_R: (B, 3, 3)
    sym_strs: list of strings
    """
    losses = []
    batch_size = pred_R.shape[0]
    device = pred_R.device

    for i in range(batch_size):
        s = sym_strs[i]
        
        # If infinite symmetry, we fall back to standard geodesic (or 0 if we want to ignore).
        # Ideally we should project, but for now let's trust PM loss for infinite objects
        # and only fix discrete ones (like lego/boxes) which cause high errors.
        if 'inf' in s:
            # For infinite symmetry (bottles, cans), we only care about the axis alignment.
            # Parse axis (default to z if not specified, but usually it is 'zinf' or similar)
            axis = torch.tensor([0., 0., 1.], device=device)
            if 'x' in s: axis = torch.tensor([1., 0., 0.], device=device)
            elif 'y' in s: axis = torch.tensor([0., 1., 0.], device=device)
             
            # Project axis: v_pred = R_pred @ axis
            v_pred = torch.matmul(pred_R[i], axis)
            v_gt = torch.matmul(gt_R[i], axis)
             
            # Angle between axes
            dot = torch.dot(v_pred, v_gt)
            dot = torch.clamp(dot, -1.0 + 1e-6, 1.0 - 1e-6)
            theta = torch.acos(dot)
            losses.append(theta)
            continue

        sym_mats = get_symmetry_matrices_torch(s, device) # (K, 3, 3)
        
        # Expand GT to all symmetric versions
        # gt_R[i]: (3,3)
        # sym_mats: (K, 3, 3)
        # targets: (K, 3, 3) -> gt_R @ sym_mat
        targets = torch.matmul(gt_R[i].unsqueeze(0), sym_mats) 
        
        # Compute geodesic loss to ALL targets
        # pred_R[i]: (3,3)
        pred = pred_R[i].unsqueeze(0).expand(len(targets), 3, 3)
        
        # R_diff = pred^T @ target
        R_diff = torch.bmm(pred.transpose(1, 2), targets)
        trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
        cos_theta = (trace - 1) / 2
        
        # Fix: Clamp strictly within (-1, 1) to avoid acos gradient explosion
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
        theta = torch.acos(cos_theta) # (K,)
        
        min_loss = torch.min(theta)
        losses.append(min_loss)

    return torch.stack(losses).mean()


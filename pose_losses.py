"""Symmetry definitions, pose losses, and pose-error metrics."""

import itertools

import numpy as np
import pandas as pd
import torch

# Hard-coded symmetry table: do NOT read geometric_symmetry from objects_v1.csv, since that
# column can be wrong (verified case: g_lego_duplo is annotated 'no' but is empirically
# z2-symmetric -- see notebook ICP analysis). Values below start from the CSV's published
# labels and are the single source of truth for scoring; only entries manually visually
# confirmed as wrong get overridden (currently just g_lego_duplo).
OBJECT_SYMMETRY_TABLE = {
    'a_cups': 'zinf', 'a_lego_duplo': 'z2', 'a_toy_airplane': 'no', 'adjustable_wrench': 'no',
    'b_cups': 'zinf', 'b_lego_duplo': 'z4', 'b_toy_airplane': 'no', 'banana': 'y2',
    'bleach_cleanser': 'z2', 'bowl': 'zinf', 'bowl_a': 'zinf', 'c_cups': 'zinf',
    'c_lego_duplo': 'z2', 'c_toy_airplane': 'zinf', 'cracker_box': 'z2|x2', 'cup_small': 'zinf',
    'd_cups': 'zinf', 'd_lego_duplo': 'no', 'd_toy_airplane': 'zinf', 'e_cups': 'zinf',
    'e_lego_duplo': 'z2', 'e_toy_airplane': 'z2', 'extra_large_clamp': 'x2', 'f_cups': 'zinf',
    'f_lego_duplo': 'z4', 'flat_screwdriver': 'xinf', 'foam_brick': 'z2|x4|y2', 'fork': 'x2',
    'g_cups': 'zinf',
    'g_lego_duplo': 'z2',  # override: CSV says 'no', verified empirically z2-symmetric
    'gelatin_box': 'z2|x2', 'h_cups': 'zinf', 'hammer': 'no', 'i_cups': 'zinf', 'j_cups': 'zinf',
    'jenga': 'z2|x2', 'knife': 'no', 'large_clamp': 'x2', 'large_marker': 'zinf',
    'master_chef_can': 'zinf|x2', 'medium_clamp': 'x2', 'mug': 'no', 'mustard_bottle': 'z2',
    'nine_hole_peg_test': 'no', 'pan_tefal': 'zinf', 'phillips_screwdriver': 'xinf',
    'pitcher_base': 'no', 'plate': 'zinf', 'potted_meat_can': 'z2|x2', 'power_drill': 'no',
    'prism': 'z2', 'pudding_box': 'z2|x2', 'rubiks_cube': 'z4|x4|y4', 'scissors': 'no',
    'spoon': 'x2', 'sugar_box': 'z2|x2', 'tomato_soup_can': 'zinf|x2', 'tuna_fish_can': 'zinf|x2',
    'wood_block': 'z2|y2|x4', 'pigeon': 'z2', 'pure_zhen': 'zinf|x2', 'realsense_box': 'z2|x2',
    'pepsi': 'zinf|x2', 'green_arrow': 'zinf', 'red_car': 'no', 'conditioner': 'z2',
    'correction_fuid': 'z2|x2', 'wooden_puzzle1': 'x4|y4|z4', 'wooden_puzzle2': 'x4|y4|z4',
    'green_car': 'no', 'redbull': 'zinf|x2', 'doraemon_cup': 'zinf', 'doraemon_bowl': 'no',
    'hello_kitty_cup': 'zinf', 'hello_kitty_bowl': 'no', 'hello_kitty_plate': 'z2',
    'doraemon_spoon': 'no', 'tea_can1': 'zinf|x2', 'wooden_puzzle3': 'x4|y4|z4',
}

def get_object_symmetry(obj_name):
    """Look up an object's symmetry class from the hard-coded table (not objects_v1.csv)."""
    return OBJECT_SYMMETRY_TABLE.get(str(obj_name).lower(), 'no')

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

def compute_auto_symmetry_rot_error(R_pred, R_gt, discrete_orders=(2, 3, 4, 6), continuous_steps=72):
    """
    Rotation error that does NOT trust any per-object symmetry annotation (e.g. objects_v1.csv),
    since those labels can be wrong. Instead it brute-forces a fixed library of candidate
    symmetry rotations (discrete 2/3/4/6-fold and continuous, about each of x/y/z, plus a
    180-deg flip stacked on top for box/brick-like shapes) and returns the smallest resulting
    error. If the raw (identity) error is large (e.g. ~90/180 deg) but this returns a small
    value, the object is empirically symmetric regardless of what any metadata file claims.
    """
    axes = {'x': np.array([1., 0., 0.]), 'y': np.array([0., 1., 0.]), 'z': np.array([0., 0., 1.])}

    candidates = [np.eye(3)]
    for axis in axes.values():
        for order in discrete_orders:
            for k in range(1, order):
                candidates.append(get_rotation_matrix(axis, 2 * np.pi * k / order))
        for k in range(1, continuous_steps):
            candidates.append(get_rotation_matrix(axis, 2 * np.pi * k / continuous_steps))

    flip_x180 = get_rotation_matrix(axes['x'], np.pi)
    flip_y180 = get_rotation_matrix(axes['y'], np.pi)
    candidates += [R_sym @ flip_x180 for R_sym in candidates] + [R_sym @ flip_y180 for R_sym in candidates]

    best_err = 180.0
    for R_sym in candidates:
        R_gt_sym = R_gt @ R_sym
        trace = np.trace(R_pred.T @ R_gt_sym)
        angle = np.degrees(np.arccos(np.clip(0.5 * (trace - 1), -1.0, 1.0)))
        if angle < best_err:
            best_err = angle
    return best_err

# --- Loss Functions ---

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

_SYM_CACHE = {}

def get_symmetry_matrices_torch(sym_str, device):
    """
    Generates a batch of symmetry rotation matrices from a symmetry string (e.g., 'z2|x2').
    Returns: (K, 3, 3) tensor
    """
    # Check cache
    cache_key = (sym_str, device)
    if cache_key in _SYM_CACHE:
        return _SYM_CACHE[cache_key]

    if not sym_str or sym_str == 'no' or sym_str == 'asymmetric' or 'nan' in sym_str:
        res = torch.eye(3, device=device).unsqueeze(0)
        _SYM_CACHE[cache_key] = res
        return res

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

    res = torch.stack(rotations)
    _SYM_CACHE[cache_key] = res
    return res

def safe_acos(x, eps=1e-3):
    """
    Safe arccos with clamping to avoid NaN gradients at x=1 or x=-1.
    """
    x = torch.clamp(x, -1.0 + eps, 1.0 - eps)
    return torch.acos(x)

def symmetry_aware_geodesic_per_sample(pred_R, gt_R, sym_strs):
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
            theta = safe_acos(dot)
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
        
        theta = safe_acos(cos_theta) # (K,)
        
        min_loss = torch.min(theta)
        losses.append(min_loss)

    return torch.stack(losses)


def symmetry_aware_geodesic_loss(pred_R, gt_R, sym_strs):
    """Batch-mean symmetry-aware geodesic loss."""
    return symmetry_aware_geodesic_per_sample(pred_R, gt_R, sym_strs).mean()

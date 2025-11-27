import os
import numpy as np
import trimesh
import open3d as o3d
import threading
import concurrent.futures
from tqdm import tqdm

# Global cache for canonical models to avoid reloading in workers
_canonical_model_cache = {}
_cache_lock = threading.Lock()

# -------------------------------------------------------------------------
# Load canonical model using CSV['location']
# -------------------------------------------------------------------------
def load_canonical_model_from_csv(obj_name, objects_df, scale):
    # Check cache first
    cache_key = (obj_name, tuple(scale) if scale is not None else None)
    with _cache_lock:
        if cache_key in _canonical_model_cache:
            return _canonical_model_cache[cache_key]

    row = objects_df[objects_df['object'] == obj_name]
    if row.empty:
        print(f"[WARN] Object {obj_name} not found in CSV.")
        return None

    model_rel_path = row.iloc[0]['location']
    model_path = os.path.join(model_rel_path, "visual_meshes", "visual.dae")

    if not os.path.exists(model_path):
        # try with leading './'
        alt_path = os.path.join(".", model_rel_path, "visual_meshes", "visual.dae")
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            print(f"[WARN] canonical mesh not found: {model_path}")
            return None

    try:
        mesh = trimesh.load(model_path, force='mesh')
        pts = mesh.sample(10000).astype(np.float32)
    except Exception as e:
        print(f"[ERROR] Failed to load {model_path}: {e}")
        return None

    # Apply scale
    if scale is not None:
        pts = pts * scale

    # Convert to Open3D PCD
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd = pcd.voxel_down_sample(0.005)  # 5mm
    pcd.estimate_normals()

    with _cache_lock:
        _canonical_model_cache[cache_key] = pcd
    return pcd


# -------------------------------------------------------------------------
# ICP refinement (Robust 3-Stage)
# -------------------------------------------------------------------------
def refine_icp(pred_R, pred_t, source_pcd, target_pcd, max_iter=50, stages=3, coarse_threshold=0.02, max_rot_change=30.0):
    # Initial Pose from PointNet
    T_init = np.eye(4, dtype=np.float32)
    T_init[:3, :3] = pred_R
    T_init[:3, 3] = pred_t

    current_pose = T_init

    try:
        # Stage 1: Coarse alignment (default 2cm threshold) - Point-to-Plane
        reg_p2l_coarse = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, coarse_threshold, current_pose,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
        )
        
        if reg_p2l_coarse.fitness > 0:
            current_pose = reg_p2l_coarse.transformation

        # Stage 2: Fine alignment (1cm threshold) - Point-to-Plane
        if stages >= 2:
            reg_p2l_fine = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, coarse_threshold * 0.5, current_pose,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
            )
            
            if reg_p2l_fine.fitness > 0:
                current_pose = reg_p2l_fine.transformation

        # Stage 3: Fine alignment (0.5cm threshold) - Point-to-Point (Fix sliding)
        if stages >= 3:
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, coarse_threshold * 0.25, current_pose,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20)
            )
            
            # Accept if RMSE improved or Fitness improved
            # We prioritize RMSE for Point-to-Point as it tightens the fit
            if reg_p2p.fitness > 0:
                 current_pose = reg_p2p.transformation

        # -------------------------------------------------------
        # SAFETY CHECK: Rotation & Translation Deviation
        # -------------------------------------------------------
        R_final = current_pose[:3, :3]
        t_final = current_pose[:3, 3]
        
        # Rotation Diff
        R_diff = R_final @ pred_R.T
        trace = np.trace(R_diff)
        cos_theta = (trace - 1) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_diff = np.degrees(np.arccos(cos_theta))

        # Translation Diff
        trans_diff = np.linalg.norm(t_final - pred_t)

        if angle_diff > max_rot_change:
            # print(f"[ICP REJECT] Rotation change {angle_diff:.2f} > {max_rot_change} deg.")
            return pred_R, pred_t, 0.0, 0.0
            
        if trans_diff > 0.01: # TIGHTENED: Max 1cm shift allowed (was 3cm)
            # print(f"[ICP REJECT] Translation change {trans_diff*100:.2f} > 1.0 cm.")
            return pred_R, pred_t, 0.0, 0.0

    except Exception as e:
        print(f"[ICP FAIL] {e}")
        return pred_R, pred_t, 0.0, 0.0

    # Get final metrics from the last successful registration
    final_fitness = 0.0
    final_rmse = 0.0
    if stages >= 3 and 'reg_p2p' in locals():
        final_fitness = reg_p2p.fitness
        final_rmse = reg_p2p.inlier_rmse
    elif stages >= 2 and 'reg_p2l_fine' in locals():
        final_fitness = reg_p2l_fine.fitness
        final_rmse = reg_p2l_fine.inlier_rmse
    elif 'reg_p2l_coarse' in locals():
        final_fitness = reg_p2l_coarse.fitness
        final_rmse = reg_p2l_coarse.inlier_rmse

    return current_pose[:3, :3], current_pose[:3, 3], final_fitness, final_rmse

# -------------------------------------------------------------------------
# Worker function for Parallel ICP
# -------------------------------------------------------------------------
def process_icp_sample(data):
    """
    Worker function to run ICP on a single sample.
    data: dict containing all necessary inputs
    """
    pred_R = data['pred_R']
    pred_t = data['pred_t']
    obs_pts = data['obs_pts']
    obj_name = data['obj_name']
    scale = data['scale']
    icp_stages = data['icp_stages']
    icp_threshold = data['icp_threshold']
    max_rot_change = data['max_rot_change']
    objects_df = data['objects_df'] # Passed as DF, but could be optimized
    
    # Load canonical model (cached inside worker process)
    source_pcd = load_canonical_model_from_csv(obj_name, objects_df, scale)
    
    if source_pcd is None:
        return pred_R, pred_t, 0.0, 0.0

    # Build target PCD
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(obs_pts)
    target_pcd = target_pcd.voxel_down_sample(0.005)
    target_pcd.estimate_normals()
    target_pcd.orient_normals_towards_camera_location(np.array([0.,0.,0.]))
    
    if len(target_pcd.points) <= 30:
        return pred_R, pred_t, 0.0, 0.0

    # Run ICP
    refined_R, refined_t, fitness, rmse = refine_icp(pred_R, pred_t, source_pcd, target_pcd, 
                                      stages=icp_stages, 
                                      coarse_threshold=icp_threshold, 
                                      max_rot_change=max_rot_change)
    
    return refined_R, refined_t, fitness, rmse

def run_icp_refinement(samples_to_process, use_icp):
    if use_icp:
        print(f"Running ICP on {len(samples_to_process)} samples using ThreadPoolExecutor...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all tasks
            futures = {executor.submit(process_icp_sample, s): i for i, s in enumerate(samples_to_process)}
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(samples_to_process), desc="ICP Refinement"):
                i = futures[future]
                try:
                    refined_R, refined_t, fitness, rmse = future.result()
                    samples_to_process[i]['pred_R'] = refined_R
                    samples_to_process[i]['pred_t'] = refined_t
                    samples_to_process[i]['icp_fitness'] = fitness
                    samples_to_process[i]['icp_rmse'] = rmse
                except Exception as e:
                    print(f"ICP Worker failed for sample {i}: {e}")
                    samples_to_process[i]['icp_fitness'] = 0.0
                    samples_to_process[i]['icp_rmse'] = 0.0
    else:
        # Initialize default values if ICP is off
        for s in samples_to_process:
            s['icp_fitness'] = 0.0
            s['icp_rmse'] = 0.0

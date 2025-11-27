import os
import torch
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import json
import pickle
import trimesh
import open3d as o3d
import concurrent.futures
import threading

import utils
import loss as loss_utils
from model import PointNet
from data import PoseDataset
import config
from matplotlib.cm import get_cmap

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_POINTS = 1024
BATCH_SIZE = 1

# Global cache for canonical models to avoid reloading in workers
_canonical_model_cache = {}
_cache_lock = threading.Lock()

# -------------------------------------------------------------------------
# Load canonical model using CSV['location'] (your working method)
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


# -------------------------------------------------------------------------
# EVALUATION
# -------------------------------------------------------------------------
def evaluate():
    args = config.get_config()
    
    use_icp = not args.no_icp
    icp_stages = args.icp_stages
    icp_threshold = args.icp_threshold
    max_rot_change = args.max_rot_change
    
    # -------------------------------
    # Load object CSV
    # -------------------------------
    objects_df = pd.read_csv(args.objects_csv)
    obj_info = {
        row['object']: {
            'dims': np.array([row['width'], row['length'], row['height']], dtype=np.float32),
            'sym': str(row['geometric_symmetry']).lower(),
            'location': row['location']
        }
        for _, row in objects_df.iterrows()
    }

    # -------------------------------
    # Load dataset and model
    # -------------------------------
    val_dataset = PoseDataset(args.split, args.training_data_dir, args.split_dir, num_points=args.num_points)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_workers)

    print("Loading PointNet...")
    model = PointNet(num_classes=args.num_classes).to(DEVICE)

    checkpoint_path = args.checkpoint_path
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print("Loading model from checkpoint dictionary...")
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("Loading model from state_dict...")
            model.load_state_dict(checkpoint)
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Using random weights.")
    
    model.eval()

    out_dir = f"{args.output_dir}/pointnet_icp" if use_icp else f"{args.output_dir}/pointnet"
    os.makedirs(out_dir, exist_ok=True)

    total_rot = total_trans = total_count = pass_count = 0
    results = {}
    current_scene = None
    scene_img = None
    
    # Store samples for parallel processing
    samples_to_process = []
    meta_list = [] # To keep track of metadata for saving results

    print(f"Starting Inference (ICP={'ON' if use_icp else 'OFF'})...")

    # ============================================================
    # Phase 1: Inference (GPU)
    # ============================================================
    for batch in tqdm(val_loader, desc="Inference"):

        pts = batch["points"].to(DEVICE)             # (1,N,9)
        centroid = batch["centroid"].numpy()[0]
        gt_R = batch["gt_rot"].numpy()[0]
        gt_t = centroid + batch["gt_t_residual"].numpy()[0]

        obj_id = batch["obj_id"].item()
        obj_dims = batch["obj_dims"].numpy()[0]
        scale = batch["scale"].numpy()[0]
        rgb_path = batch["rgb_path"][0]
        K = batch["intrinsic"].numpy()[0]
        valid_points = batch["valid_points"].item()

        # Skip if too few valid points
        if valid_points < args.min_valid_points:
            continue

        # Predict pose
        with torch.no_grad():
            outputs = model(pts, batch["obj_id"].to(DEVICE))
            if len(outputs) == 3:
                pred_rot6d, pred_t_res, _ = outputs
            else:
                pred_rot6d, pred_t_res = outputs

            pred_R = utils.rotation_6d_to_matrix(pred_rot6d)[0].cpu().numpy()
            pred_t = centroid + pred_t_res[0].cpu().numpy()

        # Prepare data for ICP/Eval
        
        # Load meta
        meta_path = rgb_path.replace("_color_kinect.png", "_meta.pkl")
        if not os.path.exists(meta_path):
            continue
        meta = pickle.load(open(meta_path, "rb"))
        
        if obj_id not in meta["object_ids"]:
            continue
        idx = list(meta["object_ids"]).index(obj_id)
        obj_name = meta["object_names"][idx]
        if obj_name not in obj_info:
            continue
        sym = obj_info[obj_name]["sym"]
        
        # Build observed points for ICP
        obs_pts = pts[0].cpu().numpy()[:, :3] + centroid
        if obs_pts.shape[0] > 8000:
            sel = np.random.choice(obs_pts.shape[0], 8000, replace=False)
            obs_pts = obs_pts[sel]

        sample_data = {
            'pred_R': pred_R,
            'pred_t': pred_t,
            'obs_pts': obs_pts,
            'obj_name': obj_name,
            'scale': scale,
            'icp_stages': icp_stages,
            'icp_threshold': icp_threshold,
            'max_rot_change': max_rot_change,
            'objects_df': objects_df,
            # Metadata for saving/metrics
            'gt_R': gt_R,
            'gt_t': gt_t,
            'sym': sym,
            'scene_name': os.path.basename(rgb_path).split("_")[0],
            'obj_id': obj_id,
            'rgb_path': rgb_path,
            'obj_dims': obj_dims,
            'K': K,
            'meta_extrinsic': meta["extrinsic"]
        }
        samples_to_process.append(sample_data)

    # ============================================================
    # Phase 2: ICP Refinement (Parallel CPU)
    # ============================================================
    refined_results = []
    
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
    
    # ============================================================
    # Phase 3: Metrics & Visualization
    # ============================================================
    print("Computing metrics and generating visualizations...")
    
    # Setup Color Palette
    NUM_OBJECTS = 79
    cmap = get_cmap('rainbow', NUM_OBJECTS)
    COLOR_PALETTE = np.array([cmap(i)[:3] for i in range(NUM_OBJECTS + 3)])
    COLOR_PALETTE = np.array(COLOR_PALETTE * 255, dtype=np.uint8)
    COLOR_PALETTE[-3] = [119, 135, 150]
    COLOR_PALETTE[-2] = [176, 194, 216]
    COLOR_PALETTE[-1] = [255, 255, 225]

    object_metrics = {}
    
    for s in tqdm(samples_to_process, desc="Metrics"):
        pred_R = s['pred_R']
        pred_t = s['pred_t']
        gt_R = s['gt_R']
        gt_t = s['gt_t']
        sym = s['sym']
        scene_name = s['scene_name']
        obj_id = s['obj_id']
        rgb_path = s['rgb_path']
        obj_dims = s['obj_dims']
        scale = s['scale']
        K = s['K']
        
        # Metrics
        if args.split == 'test':
            rot_err = 0.0
            trans_err = 0.0
            
            # Per-object metrics (ICP Fitness/RMSE)
            if obj_id not in object_metrics:
                object_metrics[obj_id] = {'fitness': [], 'rmse': [], 'name': s['obj_name']}
            object_metrics[obj_id]['fitness'].append(s['icp_fitness'])
            object_metrics[obj_id]['rmse'].append(s['icp_rmse'])
            
        else:
            rot_err = loss_utils.compute_symmetry_aware_loss(pred_R, gt_R, sym)
            trans_err = np.linalg.norm(pred_t - gt_t) * 100

            total_rot += rot_err
            total_trans += trans_err
            total_count += 1
            if rot_err < 5.0 and trans_err < 1.0:
                pass_count += 1
            
            # Per-object metrics
            if obj_id not in object_metrics:
                object_metrics[obj_id] = {'rot': [], 'trans': [], 'name': s['obj_name']}
            object_metrics[obj_id]['rot'].append(rot_err)
            object_metrics[obj_id]['trans'].append(trans_err)
            
        # Save Results
        if scene_name not in results:
            results[scene_name] = {"poses_world": [None] * 80}

        T_wc = np.array(s['meta_extrinsic']).reshape(4, 4)
        T_co = np.eye(4)
        T_co[:3, :3] = pred_R
        T_co[:3, 3] = pred_t
        T_ow = np.linalg.inv(T_wc) @ T_co
        results[scene_name]["poses_world"][obj_id] = T_ow.tolist()
        
        # Visualization (Sequential for now to avoid race conditions on image writing)
        if scene_name != current_scene:
            if scene_img is not None:
                cv2.imwrite(f"{out_dir}/{current_scene}.png", scene_img)
            
            # Load and Upscale (2x) for higher resolution
            original_img = cv2.imread(rgb_path)
            scene_img = cv2.resize(original_img, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            current_scene = scene_name
            
        if scene_img is not None:
            size = obj_dims * scale
            
            # Determine Color and Thickness based on Error
            # High Error (Rot > 5 deg OR Trans > 1 cm) -> RED
            if args.split != 'test' and (rot_err > 5.0 or trans_err > 1.0):
                color = (0, 0, 255) # Red (BGR)
                thickness = 3
                print(f"[FAIL] Scene: {scene_name} | Obj: {s['obj_name']} | Rot: {rot_err:.1f} deg, Trans: {trans_err:.1f} cm")
            else:
                # Pass -> Green (BGR)
                color = (0, 255, 0)
                thickness = 2
            
            # Scaled Intrinsics
            K_scaled = K.copy()
            K_scaled[0, 0] *= 2
            K_scaled[1, 1] *= 2
            K_scaled[0, 2] *= 2
            K_scaled[1, 2] *= 2

            utils.draw_projected_box3d(scene_img, pred_t, size, pred_R,
                                       np.eye(4), K_scaled, color=color, thickness=thickness)

    # Save last scene
    if scene_img is not None:
        cv2.imwrite(f"{out_dir}/{current_scene}.png", scene_img)

    # Summary
    print(f"\nEval finished.")
    
    csv_rows = []
    
    if args.split == 'test':
        print("\n" + "="*70)
        print(f"{'Object Name':<30} | {'Avg Fitness':<15} | {'Avg RMSE':<15}")
        print("-" * 70)
        
        for oid in sorted(object_metrics.keys()):
            m = object_metrics[oid]
            avg_fitness = np.mean(m['fitness'])
            avg_rmse = np.mean(m['rmse'])
            print(f"{m['name']:<30} | {avg_fitness:<15.4f} | {avg_rmse:<15.4f}")
            
            csv_rows.append({
                "object_id": oid,
                "object_name": m['name'],
                "avg_icp_fitness": avg_fitness,
                "avg_icp_rmse": avg_rmse,
                "sample_count": len(m['fitness'])
            })
        print("="*70 + "\n")
        
        csv_filename = "test_metrics_icp.csv" if use_icp else "test_metrics.csv"

    else:
        print(f"Pass: {pass_count}/{total_count} ({100*pass_count/total_count:.2f}%)")
        print(f"Avg Rot Err: {total_rot/total_count:.2f} deg")
        print(f"Avg Trans Err: {total_trans/total_count:.2f} cm")
        
        # Per-Object Table
        print("\n" + "="*70)
        print(f"{'Object Name':<30} | {'Rot Err (deg)':<15} | {'Trans Err (cm)':<15}")
        print("-" * 70)
        
        for oid in sorted(object_metrics.keys()):
            m = object_metrics[oid]
            avg_rot = np.mean(m['rot'])
            avg_trans = np.mean(m['trans'])
            print(f"{m['name']:<30} | {avg_rot:<15.2f} | {avg_trans:<15.2f}")
            
            csv_rows.append({
                "object_id": oid,
                "object_name": m['name'],
                "avg_rot_err_deg": avg_rot,
                "avg_trans_err_cm": avg_trans,
                "sample_count": len(m['rot'])
            })
        print("="*70 + "\n")

        csv_filename = "error_metrics_icp.csv" if use_icp else "error_metrics.csv"

    # Save CSV
    csv_path = os.path.join(args.output_dir, csv_filename)
    try:
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        print(f"Saved per-object metrics to {csv_path}")
    except PermissionError:
        print(f"[ERROR] Permission denied for {csv_path}. File is likely open in Excel/Viewer.")
        csv_path_backup = os.path.join(args.output_dir, f"backup_{csv_filename}")
        pd.DataFrame(csv_rows).to_csv(csv_path_backup, index=False)
        print(f"Saved to {csv_path_backup} instead.")

    out_json = "pointnet_predictions_icp.json" if use_icp else "pointnet_predictions.json"
    json.dump(results, open(out_json, "w"), indent=2)


if __name__ == "__main__":
    evaluate()

import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import json
import argparse
from PIL import Image
import pickle
import utils
import cv2
# Set Ray environment variable to suppress warning
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
import ray
from tqdm import tqdm
import open3d as o3d
import trimesh
import random
import pandas as pd
import benchmark_utils.pose_utils as pose_utils

training_data_dir = "./training_data_filtered/training_data/v2.2"
split_dir = "./training_data_filtered/training_data/splits/v2"

def load_pile(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def get_split_files(split_name):
    with open(os.path.join(split_dir, f"{split_name}.txt"), 'r') as f:
        prefix = [os.path.join(training_data_dir, line.strip()) for line in f if line.strip()]
        rgb = [p + "_color_kinect.png" for p in prefix]
        depth = [p + "_depth_kinect.png" for p in prefix]
        label = [p + "_label_kinect.png" for p in prefix]
        meta = [p + "_meta.pkl" for p in prefix]
    return rgb, depth, label, meta

# create dataset class
class ICPDataset:
    def __init__(self, split_name):
        self.rgb_files, self.depth_files, self.label_files, self.meta_files = get_split_files(split_name)
        self.length = len(self.rgb_files)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rgb = np.asarray(Image.open(self.rgb_files[idx]), dtype=np.float32) / 255.0
        depth = np.asarray(Image.open(self.depth_files[idx]), dtype=np.float32) / 1000.0
        label = np.asarray(Image.open(self.label_files[idx]), dtype=np.int32)
        with open(self.meta_files[idx], "rb") as f:
            meta = pickle.load(f)
        name = os.path.basename(self.meta_files[idx]).replace("_meta.pkl", "")
        return rgb, depth, label, meta, name
    
# lift depth to point cloud
def depth_to_point_cloud(depth, meta):
    intrinsic = meta['intrinsic']
    z = depth
    v, u = np.indices(z.shape)
    uv1 = np.stack([u + 0.5, v + 0.5, np.ones_like(z)], axis=-1)
    points_viewer = uv1 @ np.linalg.inv(intrinsic).T * z[..., None]  # [H, W, 3]
    return points_viewer

# icp algorithm
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

def compute_min_symmetry_loss(R_pred, R_gt, sym_str):
    sym_axes, sym_orders = parse_geometric_symmetry(sym_str)
    sym_rots, rot_axis = pose_utils.get_symmetry_rotations(sym_axes, sym_orders)
    
    losses = []
    
    # If infinite symmetry (e.g. cylinder)
    if rot_axis is not None:
        # For infinite symmetry, we care about the alignment of the axis
        # But we also have discrete symmetries (like flip) in sym_rots
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

@ray.remote
def process_scene(idx, split_name, objects_df, verbose):
    # Define Color Palette inside worker
    NUM_OBJECTS = 79
    cmap = plt.get_cmap('rainbow', NUM_OBJECTS)
    COLOR_PALETTE = np.array([cmap(i)[:3] for i in range(NUM_OBJECTS + 3)])
    COLOR_PALETTE = np.array(COLOR_PALETTE * 255, dtype=np.uint8)
    COLOR_PALETTE[-3] = [119, 135, 150]
    COLOR_PALETTE[-2] = [176, 194, 216]
    COLOR_PALETTE[-1] = [255, 255, 225]

    try:
        dataset = ICPDataset(split_name)
        rgb, depth, label, meta, name = dataset[idx]
    except Exception as e:
        return None, None, 0, 0, [f"Error loading scene {idx}: {e}"], 0.0, 0.0

    scene_results = {"poses_world": [None] * NUM_OBJECTS}
    logs = []
    pass_count = 0
    total_count = 0
    scene_rot_err = 0.0
    scene_trans_err = 0.0

    # Prepare visualization image
    img_vis = COLOR_PALETTE[label]
    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)

    # Camera intrinsics
    intrinsics = meta['intrinsic']
    if isinstance(intrinsics, dict):
        K = np.array([
            [intrinsics['fx'], 0, intrinsics['cx']],
            [0, intrinsics['fy'], intrinsics['cy']],
            [0, 0, 1]
        ])
    else:
        K = np.array(intrinsics)

    # Camera extrinsics (World to Camera)
    T_wc = np.array(meta['extrinsic']).reshape(4, 4)
    
    # Generate scene point cloud
    scene_points_map = depth_to_point_cloud(depth, meta)
    
    object_ids = meta['object_ids']
    object_names = meta['object_names']
    
    for i, obj_id in enumerate(object_ids):
        # Mask for the object
        mask = (label == obj_id)
        if not np.any(mask):
            continue
            
        # Extract object points from scene (Target)
        target_points = scene_points_map[mask]
        if len(target_points) < 100:
            continue
            
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_points)
        
        # Voxel Downsample to get uniform points (better for corners)
        target_pcd = target_pcd.voxel_down_sample(voxel_size=0.005) # 5mm voxel
        
        target_pcd.estimate_normals()
        target_pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
        
        # Identify object name and load mesh
        if i < len(object_names):
            obj_name = object_names[i]
        else:
            # Fallback to CSV lookup by ID
            csv_idx = obj_id - 1
            if csv_idx < 0 or csv_idx >= len(objects_df):
                csv_idx = obj_id - 2
            if csv_idx >= 0 and csv_idx < len(objects_df):
                obj_name = objects_df.iloc[csv_idx]['object']
            else:
                logs.append(f"Could not determine name for Obj ID {obj_id}")
                continue

        # Find row in CSV
        row = objects_df[objects_df['object'] == obj_name]
        if row.empty:
            logs.append(f"Object {obj_name} not found in CSV.")
            continue
        
        model_rel_path = row.iloc[0]['location']
        sym_str = row.iloc[0]['geometric_symmetry']
        
        # Construct full path
        model_path = os.path.join(model_rel_path, "visual_meshes", "visual.dae")
        if not os.path.exists(model_path):
            if os.path.exists(os.path.join(".", model_rel_path)):
                    model_path = os.path.join(".", model_rel_path, "visual_meshes", "visual.dae")
            else:
                logs.append(f"Model file not found: {model_path}")
                continue
            
        try:
            mesh = trimesh.load(model_path, force='mesh')
            # Sample more points then downsample for uniform coverage
            source_points = mesh.sample(10000)
            
            # Apply Scale
            if meta['scales'][obj_id] is not None:
                scale = meta['scales'][obj_id]
                source_points = source_points * scale
            
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(source_points)
            source_pcd = source_pcd.voxel_down_sample(voxel_size=0.005) # 5mm voxel
            source_pcd.estimate_normals()
        except Exception as e:
            logs.append(f"Failed to load mesh {model_path}: {e}")
            continue

        # Ground Truth Pose (Object to World)
        if meta['poses_world'][obj_id] is None:
            continue
        T_ow = np.array(meta['poses_world'][obj_id]).reshape(4, 4)
        
        # GT Pose in Camera Frame (Object to Camera)
        T_co_gt = T_wc @ T_ow
        
        # Initial Pose for ICP
        init_pose = T_co_gt
        
        # Evaluate Initial Pose
        eval_init = o3d.pipelines.registration.evaluate_registration(
            source_pcd, target_pcd, 0.02, init_pose
        )
        if verbose:
            logs.append(f"  Init Pose Fitness: {eval_init.fitness:.4f}, RMSE: {eval_init.inlier_rmse:.4f}")

        # Run ICP (Point-to-Plane)
        try:
            # Stage 1: Coarse alignment (2cm threshold)
            reg_p2l_coarse = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, 0.02, init_pose,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
            )
            if verbose:
                logs.append(f"  Stage 1 Fitness: {reg_p2l_coarse.fitness:.4f}, RMSE: {reg_p2l_coarse.inlier_rmse:.4f}")
            
            if reg_p2l_coarse.fitness == 0:
                    if verbose: logs.append("  Stage 1 failed (0 fitness), reverting to Init Pose")
                    current_pose = init_pose
            else:
                    current_pose = reg_p2l_coarse.transformation

            # Stage 2: Fine alignment (1cm threshold)
            reg_p2l_fine = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, 0.01, current_pose,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
            )
            if verbose:
                logs.append(f"  Stage 2 Fitness: {reg_p2l_fine.fitness:.4f}, RMSE: {reg_p2l_fine.inlier_rmse:.4f}")
            
            if reg_p2l_fine.fitness > 0:
                current_pose = reg_p2l_fine.transformation
            else:
                if verbose: logs.append("  Stage 2 failed (0 fitness), keeping Stage 1/Init")
            
            # Stage 3: Point-to-Point Fine alignment (1cm threshold) to fix sliding
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, 0.01, current_pose,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
            )
            if verbose:
                logs.append(f"  Stage 3 (P2P) Fitness: {reg_p2p.fitness:.4f}, RMSE: {reg_p2p.inlier_rmse:.4f}")

            if reg_p2p.fitness > 0 and reg_p2p.inlier_rmse < reg_p2l_fine.inlier_rmse:
                    final_pose = reg_p2p.transformation
            elif reg_p2l_fine.fitness > 0:
                    final_pose = reg_p2l_fine.transformation
            else:
                    final_pose = current_pose

        except Exception as e:
            logs.append(f"ICP failed: {e}")
            continue
        
        # Compute Errors
        R_pred = final_pose[:3, :3]
        R_gt = T_co_gt[:3, :3]
        
        # Symmetry aware rotation error
        rre_deg = compute_min_symmetry_loss(R_pred, R_gt, sym_str)
        
        t_pred = final_pose[:3, 3]
        t_gt = T_co_gt[:3, 3]
        rte = pose_utils.compute_rte(t_pred, t_gt) # Returns meters
        rte_cm = rte * 100
        
        # Store result
        T_ow_pred = np.linalg.inv(T_wc) @ final_pose
        scene_results["poses_world"][obj_id] = T_ow_pred.tolist()
        
        if verbose:
            logs.append(f"Scene {name}, Obj {obj_name} (ID {obj_id}): Rot Err = {rre_deg:.2f} deg, Trans Err = {rte_cm:.2f} cm")
        
        total_count += 1
        scene_rot_err += rre_deg
        scene_trans_err += rte_cm

        if rre_deg < 20.0 and rte_cm < 2.0:
            if verbose: logs.append("  [PASS]")
            pass_count += 1
        else:
            if verbose: logs.append("  [FAIL]")

        # Draw bounding box
        try:
            size = mesh.extents
            if meta['scales'][obj_id] is not None:
                scale = meta['scales'][obj_id]
                size = size * scale
            
            # Draw Predicted Box in Blue (using final_pose directly in Camera Frame)
            utils.draw_projected_box3d(img_vis, final_pose[:3, 3], size, final_pose[:3, :3], np.eye(4), K, color=None)
        except Exception as e:
            logs.append(f"Failed to draw box: {e}")
        
    # Save visualization
    os.makedirs("output_images", exist_ok=True)
    # Save with higher DPI using matplotlib
    plt.figure(figsize=(10, 8), dpi=300)
    plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.savefig(f"output_images/{name}_pred.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    return name, scene_results, pass_count, total_count, logs, scene_rot_err, scene_trans_err

def run_icp_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true', help="Print detailed logs")
    parser.add_argument('--num_scenes', type=int, default=None, help="Number of scenes to evaluate")
    parser.add_argument('--no_ray', action='store_true', help="Disable Ray for debugging")
    args = parser.parse_args()

    # Load object mapping
    objects_csv_path = "benchmark_utils/objects_v1.csv"
    if not os.path.exists(objects_csv_path):
        print(f"Error: {objects_csv_path} not found.")
        return

    objects_df = pd.read_csv(objects_csv_path)
    
    # Load dataset
    split_name = 'val'
    
    try:
        dataset = ICPDataset(split_name)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if args.num_scenes:
        indices = range(min(args.num_scenes, len(dataset)))
    else:
        indices = range(len(dataset))
    
    print(f"Selected {len(indices)} scenes for evaluation from split '{split_name}'.")
    
    # Clean up old images
    if os.path.exists("output_images"):
        shutil.rmtree("output_images")
    os.makedirs("output_images", exist_ok=True)
    
    if not args.no_ray:
        ray.init()
    
    results = {}
    total_pass = 0
    total_objects = 0
    total_rot_err = 0.0
    total_trans_err = 0.0

    if not args.no_ray:
        # Ray Parallel Execution
        futures = [process_scene.remote(idx, split_name, objects_df, args.verbose) for idx in indices]
        
        for _ in tqdm(range(len(futures)), desc="Processing Scenes"):
            done_id, futures = ray.wait(futures)
            name, scene_res, p_count, t_count, logs, s_rot, s_trans = ray.get(done_id[0])
            
            if name:
                results[name] = scene_res
                total_pass += p_count
                total_objects += t_count
                total_rot_err += s_rot
                total_trans_err += s_trans
                if args.verbose and logs:
                    print(f"\n--- Logs for {name} ---")
                    for log in logs:
                        print(log)
    else:
        # Serial Execution (for debugging)
        # We need a wrapper to call the logic without ray.remote
        # But since process_scene is decorated, we can't call it directly easily unless we extract logic.
        # Or we can use .remote() and ray.get() sequentially.
        print("Running in serial mode (using Ray sequentially)...")
        ray.init(local_mode=True)
        for idx in tqdm(indices, desc="Processing Scenes"):
            name, scene_res, p_count, t_count, logs, s_rot, s_trans = ray.get(process_scene.remote(idx, split_name, objects_df, args.verbose))
            if name:
                results[name] = scene_res
                total_pass += p_count
                total_objects += t_count
                total_rot_err += s_rot
                total_trans_err += s_trans
                if args.verbose and logs:
                    print(f"\n--- Logs for {name} ---")
                    for log in logs:
                        print(log)

    # Save JSON
    with open("icp_predictions.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSummary: {total_pass}/{total_objects} passed.")
    if total_objects > 0:
        print(f"Average Rotation Error: {total_rot_err / total_objects:.2f} deg")
        print(f"Average Translation Error: {total_trans_err / total_objects:.2f} cm")

if __name__ == "__main__":
    run_icp_evaluation()

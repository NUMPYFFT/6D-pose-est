import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import json
import pickle
import utils
import itertools
from model import PointNet
from data import PoseDataset

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1 # Process one by one for evaluation to handle different objects/scenes easily
NUM_POINTS = 1024

# --- Evaluation ---
def evaluate():
    # Load Object Info for Symmetry and Dimensions
    objects_df = pd.read_csv("benchmark_utils/objects_v1.csv")
    obj_info_map = {}
    for _, row in objects_df.iterrows():
        obj_name = str(row['object']).strip()
        sym = str(row['geometric_symmetry']).lower()
        
        # Manual Overrides (Same as in data.py)
        # if obj_name in ['g_lego_duplo', 'e_lego_duplo', 'nine_hole_peg_test']: sym = 'z2|x2|y2'
        # if obj_name in ['mustard_bottle', 'bleach_cleanser']: sym = 'z2'
        # if obj_name == 'extra_large_clamp': sym = 'x2|z2'
        
        obj_info_map[obj_name] = {
            'sym': sym,
            'dims': np.array([row['width'], row['length'], row['height']], dtype=np.float32)
        }

    # Load Dataset
    val_dataset = PoseDataset("val", num_points=NUM_POINTS, subset_size=None)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Load Model
    model = PointNet().to(DEVICE)
    model.load_state_dict(torch.load("pointnet_best.pth"))
    model.eval()
    
    print(f"Evaluating on {len(val_dataset)} samples...")
    
    results = {}
    total_rot_err = 0.0
    total_trans_err = 0.0
    pass_count = 0
    total_count = 0
    
    # Create output dir
    os.makedirs("output_images_pointnet", exist_ok=True)
    
    current_scene_name = ""
    current_scene_img = None
    visualize_this_scene = False

    for i, batch in enumerate(tqdm(val_loader)):
        points = batch['points'].to(DEVICE)
        gt_rot = batch['gt_rot'].to(DEVICE)
        gt_t_residual = batch['gt_t_residual'].to(DEVICE)
        centroid = batch['centroid'].to(DEVICE)
        obj_ids = batch['obj_id'].to(DEVICE)
        rgb_paths = batch['rgb_path']
        intrinsics = batch['intrinsic'].numpy()
        scales = batch['scale']
        
        with torch.no_grad():
            pred_rot_6d, pred_trans_residual = model(points, obj_ids)
            pred_R_batch = utils.rotation_6d_to_matrix(pred_rot_6d)
            
        # Process batch (size 1)
        for b in range(len(points)):
            obj_id = obj_ids[b].item()
            rgb_path = rgb_paths[b]
            
            # Scene Management
            scene_name = os.path.basename(rgb_path).split("_")[0]
            
            if scene_name != current_scene_name:
                # Save previous scene image if it exists
                if current_scene_img is not None and visualize_this_scene:
                    cv2.imwrite(f"output_images_pointnet/{current_scene_name}_all.png", current_scene_img)
                
                # Start new scene
                current_scene_name = scene_name
                visualize_this_scene = True
                
                if visualize_this_scene:
                    current_scene_img = cv2.imread(rgb_path)
                else:
                    current_scene_img = None

            # Get Object Name
            # We need to open the meta file to get the name map or use a global map
            # Since we don't have the meta file path directly in the batch (only rgb_path),
            # we can infer it.
            meta_path = rgb_path.replace("_color_kinect.png", "_meta.pkl")
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            
            if obj_id in meta['object_ids']:
                idx_list = list(meta['object_ids']).index(obj_id)
                obj_name = meta['object_names'][idx_list]
            else:
                print(f"MISSING: Scene {scene_name} | ObjID {obj_id} not in meta")
                continue
                
            if obj_name not in obj_info_map:
                print(f"MISSING: Scene {scene_name} | Object {obj_name} not in CSV")
                continue
                
            info = obj_info_map[obj_name]
            sym_str = info['sym']
            dims = info['dims']
            
            # Get Predictions
            pred_R = pred_R_batch[b].cpu().numpy()
            pred_t_res = pred_trans_residual[b].cpu().numpy()
            c = centroid[b].cpu().numpy()
            
            pred_t = c + pred_t_res
            
            # Get GT
            gt_R_np = gt_rot[b].cpu().numpy()
            gt_t_res_np = gt_t_residual[b].cpu().numpy()
            gt_t = c + gt_t_res_np
            
            # Compute Metrics
            # Use the same logic as training: if symmetric, use Chamfer-like metric or 0 error if close enough?
            # Actually, for evaluation we usually want the "Symmetry Aware Rotation Error".
            # The function compute_symmetry_aware_loss in utils.py calculates the angular distance 
            # to the closest symmetric pose. This is correct.
            
            rot_err = utils.compute_symmetry_aware_loss(pred_R, gt_R_np, sym_str)
            
            # However, for infinite symmetry (bottles, bowls), the error might still be high if the axis is slightly off.
            # Let's double check if the sym_str is being parsed correctly.
            # In data.py we added manual overrides. We should do the same here or rely on the CSV.
            # The CSV has 'zinf' for bowls.
            
            trans_err = np.linalg.norm(pred_t - gt_t) * 100 # cm
            
            total_rot_err += rot_err
            total_trans_err += trans_err
            total_count += 1
            
            if rot_err < 5.0 and trans_err < 1.0:
                pass_count += 1
                
            # Save Result
            if scene_name not in results:
                results[scene_name] = {"poses_world": [None] * 80} # Max ID approx 80
            
            # We need T_ow (Object to World). 
            # We have T_co (Object to Camera) = [pred_R | pred_t]
            # T_co = T_wc @ T_ow  =>  T_ow = inv(T_wc) @ T_co
            T_wc = np.array(meta['extrinsic']).reshape(4, 4)
            T_co_pred = np.eye(4)
            T_co_pred[:3, :3] = pred_R
            T_co_pred[:3, 3] = pred_t
            
            T_ow_pred = np.linalg.inv(T_wc) @ T_co_pred
            results[scene_name]["poses_world"][obj_id] = T_ow_pred.tolist()
            
            # Visualization
            if visualize_this_scene and current_scene_img is not None:
                K = intrinsics[b]
                
                # Scale
                scale = scales[b].numpy()
                size = dims * scale
                
                # Draw GT (Blue)
                # utils.draw_projected_box3d(current_scene_img, gt_t, size, gt_R_np, np.eye(4), K, color=(255, 0, 0), thickness=1)
                
                # Draw Pred (Green)
                utils.draw_projected_box3d(current_scene_img, pred_t, size, pred_R, np.eye(4), K, color=(0, 255, 0), thickness=2)
                
                # Draw Axes
                # utils.draw_axes(current_scene_img, pred_t, pred_R, np.eye(4), K, length=max(dims)/2, thickness=2)
                
    # Save the last scene image if needed
    if current_scene_img is not None and visualize_this_scene:
        cv2.imwrite(f"output_images_pointnet/{current_scene_name}_all.png", current_scene_img)

    print(f"Evaluation Complete.")
    print(f"Pass Rate: {pass_count}/{total_count} ({pass_count/total_count*100:.2f}%)")
    print(f"Avg Rotation Error: {total_rot_err/total_count:.2f} deg")
    print(f"Avg Translation Error: {total_trans_err/total_count:.2f} cm")
    
    with open("pointnet_predictions.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    evaluate()

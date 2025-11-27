import os
import torch
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import pickle

import utils
import loss as loss_utils
from model import PointNet
from data import PoseDataset
import config
import icp
from matplotlib.cm import get_cmap

# -------------------------------------------------------------------------
# Inference Loop
# -------------------------------------------------------------------------
def run_inference(model, val_loader, args, obj_info):
    samples_to_process = []
    print(f"Starting Inference...")
    
    device = torch.device(args.device)

    for batch in tqdm(val_loader, desc="Inference"):
        pts = batch["points"].to(device)             # (B,N,9)
        
        # Predict pose for the whole batch
        with torch.no_grad():
            outputs = model(pts, batch["obj_id"].to(device))
            if len(outputs) == 3:
                pred_rot6d, pred_t_res, _ = outputs
            else:
                pred_rot6d, pred_t_res = outputs

            # Convert rotation for the whole batch
            pred_Rs = utils.rotation_6d_to_matrix(pred_rot6d).cpu().numpy() # (B, 3, 3)
            pred_t_res_numpy = pred_t_res.cpu().numpy() # (B, 3)

        # Iterate over batch elements
        batch_size = pts.shape[0]
        
        centroids = batch["centroid"].numpy()
        gt_rots = batch["gt_rot"].numpy()
        gt_t_residuals = batch["gt_t_residual"].numpy()
        obj_ids = batch["obj_id"].numpy()
        obj_dims_batch = batch["obj_dims"].numpy()
        scales = batch["scale"].numpy()
        rgb_paths = batch["rgb_path"]
        Ks = batch["intrinsic"].numpy()
        valid_points_batch = batch["valid_points"].numpy()

        for i in range(batch_size):
            valid_points = valid_points_batch[i]
            # Skip if too few valid points
            if valid_points < args.min_valid_points:
                continue

            centroid = centroids[i]
            gt_R = gt_rots[i]
            gt_t = centroid + gt_t_residuals[i]
            
            obj_id = obj_ids[i]
            obj_dims = obj_dims_batch[i]
            scale = scales[i]
            rgb_path = rgb_paths[i]
            K = Ks[i]
            
            pred_R = pred_Rs[i]
            pred_t = centroid + pred_t_res_numpy[i]

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
            # pts is (B, N, 9), so pts[i] is (N, 9)
            obs_pts = pts[i].cpu().numpy()[:, :3] + centroid
            if obs_pts.shape[0] > 8000:
                sel = np.random.choice(obs_pts.shape[0], 8000, replace=False)
                obs_pts = obs_pts[sel]

            sample_data = {
                'pred_R': pred_R,
                'pred_t': pred_t,
                'obs_pts': obs_pts,
                'obj_name': obj_name,
                'scale': scale,
                'icp_stages': args.icp_stages,
                'icp_threshold': args.icp_threshold,
                'max_rot_change': args.max_rot_change,
                'objects_df': pd.read_csv(args.objects_csv), # Re-read or pass? Better pass.
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
        
    return samples_to_process

def compute_metrics_and_visualize(samples_to_process, args, out_dir):
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
    current_scene = None
    scene_img = None
    
    total_rot = total_trans = total_count = pass_count = 0
    csv_rows = []

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
        
    return csv_rows

# -------------------------------------------------------------------------
# EVALUATION
# -------------------------------------------------------------------------
def evaluate():
    args = config.get_config()
    
    use_icp = not args.no_icp
    
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
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("Loading PointNet...")
    device = torch.device(args.device)
    model = PointNet(num_classes=args.num_classes).to(device)

    checkpoint_path = args.checkpoint_path
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
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

    # Phase 1: Inference
    samples_to_process = run_inference(model, val_loader, args, obj_info)
    
    # Phase 2: ICP Refinement
    icp.run_icp_refinement(samples_to_process, use_icp)
    
    # Phase 3: Metrics & Visualization
    csv_rows = compute_metrics_and_visualize(samples_to_process, args, out_dir)

    # Save CSV
    csv_filename = "test_metrics_icp.csv" if use_icp and args.split == 'test' else \
                   "test_metrics.csv" if args.split == 'test' else \
                   "error_metrics_icp.csv" if use_icp else "error_metrics.csv"
                   
    csv_path = os.path.join(args.output_dir, csv_filename)
    try:
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        print(f"Saved per-object metrics to {csv_path}")
    except PermissionError:
        print(f"[ERROR] Permission denied for {csv_path}. File is likely open in Excel/Viewer.")
        csv_path_backup = os.path.join(args.output_dir, f"backup_{csv_filename}")
        pd.DataFrame(csv_rows).to_csv(csv_path_backup, index=False)
        print(f"Saved to {csv_path_backup} instead.")


if __name__ == "__main__":
    evaluate()

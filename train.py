import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import utils
from data import PoseDataset
from model import PointNet

NUM_POINTS = 4096
BATCH_SIZE = 128
LR = 0.001
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # 1. Setup Datasets
    train_dataset = PoseDataset("train", num_points=NUM_POINTS, subset_size=None)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    val_dataset = PoseDataset("val", num_points=NUM_POINTS, subset_size=None)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Prepare visualization indices (Last Scene)
    vis_scene_idx = val_dataset.samples[-1][0]
    vis_indices = [i for i, (s_idx, _) in enumerate(val_dataset.samples) if s_idx == vis_scene_idx]
    print(f"Visualization Scene Index: {vis_scene_idx}, Objects: {len(vis_indices)}")

    # 2. Setup Model
    model = PointNet().to(DEVICE)
    
    # Restart from scratch (commented out loading)
    # if os.path.exists("pointnet_best.pth"): ...

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)

    pm_loss_fn = utils.point_matching_loss
    l1 = nn.L1Loss()

    best_val_rot = 99999.0

    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        total_loss = 0
        total_trans_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            pts = batch['points'].to(DEVICE)
            gt_R = batch['gt_rot'].to(DEVICE)
            gt_t = batch['gt_t_residual'].to(DEVICE)
            obj_id = batch['obj_id'].to(DEVICE)
            sym = batch['sym_str']
            xyz = pts[:,:,:3]

            opt.zero_grad()

            pred_rot6d, pred_t = model(pts, obj_id)
            pred_R = utils.rotation_6d_to_matrix(pred_rot6d)

            loss_pm = pm_loss_fn(pred_R, pred_t, gt_R, gt_t, xyz, sym)
            loss_t = l1(pred_t, gt_t)
            loss_geo = utils.symmetry_aware_geodesic_loss(pred_R, gt_R, sym)

            loss = 100*loss_pm + 5.0*loss_t + 10.0 * loss_geo

            loss.backward()
            opt.step()
            total_loss += loss.item()
            total_trans_loss += loss_t.item()

        scheduler.step()
        
        # --- VALIDATION ---
        model.eval()
        val_geo_loss = 0.0
        val_trans_err_cm = 0.0
        val_count = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                pts = batch['points'].to(DEVICE)
                gt_R = batch['gt_rot'].to(DEVICE)
                gt_t = batch['gt_t_residual'].to(DEVICE)
                obj_id = batch['obj_id'].to(DEVICE)
                sym = batch['sym_str']
                
                pred_rot6d, pred_t = model(pts, obj_id)
                pred_R = utils.rotation_6d_to_matrix(pred_rot6d)
                
                # Rotation Error (Geodesic)
                loss_g = utils.symmetry_aware_geodesic_loss(pred_R, gt_R, sym)
                val_geo_loss += loss_g.item() * pts.shape[0]
                
                # Translation Error (L2 Distance in cm)
                # pred_t and gt_t are residuals here
                trans_dist = torch.norm(pred_t - gt_t, dim=1) * 100.0 # cm
                val_trans_err_cm += trans_dist.sum().item()
                
                val_count += pts.shape[0]

        avg_val_rot_deg = (val_geo_loss / val_count) * (180.0 / np.pi)
        avg_val_trans_cm = val_trans_err_cm / val_count
        
        print(f"Epoch {epoch+1} | Train Loss {total_loss/len(train_loader):.4f} | Val Rot Error {avg_val_rot_deg:.2f} deg | Val Trans Error {avg_val_trans_cm:.2f} cm")

        # Save Best
        if avg_val_rot_deg < best_val_rot:
            best_val_rot = avg_val_rot_deg
            torch.save(model.state_dict(), "pointnet_best.pth")
            print("Saved best model (Val Improvement)")

        # --- VISUALIZATION (Full Scene) ---
        os.makedirs("training_vis", exist_ok=True)
        
        # Load the scene image (just once)
        vis_img_path = val_dataset.rgb_files[vis_scene_idx]
        vis_img = cv2.imread(vis_img_path)
        
        with torch.no_grad():
            for idx in vis_indices:
                sample = val_dataset[idx]
                pts = sample['points'].unsqueeze(0).to(DEVICE)
                obj_id = torch.tensor([sample['obj_id']]).to(DEVICE)
                K = sample['intrinsic'] # numpy
                dims = sample['obj_dims'] # numpy
                scale = sample['scale'].numpy()
                c = sample['centroid'].numpy()
                
                pred_rot6d, pred_t_res = model(pts, obj_id)
                pred_R = utils.rotation_6d_to_matrix(pred_rot6d)[0].cpu().numpy()
                pred_t = c + pred_t_res[0].cpu().numpy()
                
                size = dims * scale
                utils.draw_projected_box3d(vis_img, pred_t, size, pred_R, np.eye(4), K, color=(0, 255, 0), thickness=2)
        
        cv2.imwrite(f"training_vis/epoch_{epoch+1}_scene.png", vis_img)

if __name__ == "__main__":
    train()

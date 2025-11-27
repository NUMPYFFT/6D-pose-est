import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import Counter
import cv2
import utils
import loss as loss_utils
from data import PoseDataset
from model import PointNet
import config

def train_one_epoch(model, loader, opt, device, args, pm_loss_fn, l1_loss_fn):
    model.train()
    total_loss = 0
    total_pm_loss = 0
    total_geo_loss = 0
    total_trans_loss = 0
    
    for batch in tqdm(loader, desc="Train"):
        pts = batch['points'].to(device)
        gt_R = batch['gt_rot'].to(device)
        gt_t = batch['gt_t_residual'].to(device)
        obj_id = batch['obj_id'].to(device)
        sym = batch['sym_str']
        xyz = pts[:,:,:3]

        opt.zero_grad()

        with torch.cuda.amp.autocast():
            pred_rot6d, pred_t = model(pts, obj_id)
            pred_R = utils.rotation_6d_to_matrix(pred_rot6d)

            loss_pm = pm_loss_fn(pred_R, pred_t, gt_R, gt_t, xyz, sym)
            loss_t = l1_loss_fn(pred_t, gt_t)
            loss_geo = loss_utils.symmetry_aware_geodesic_loss(pred_R, gt_R, sym)

            loss = args.w_pm * loss_pm + args.w_geo * loss_geo + args.w_trans * loss_t
        
        loss.backward()
        opt.step()
        
        total_loss += loss.item()
        total_pm_loss += loss_pm.item()
        total_geo_loss += loss_geo.item()
        total_trans_loss += loss_t.item()

    avg_loss = total_loss / len(loader)
    avg_pm = total_pm_loss / len(loader)
    avg_geo = total_geo_loss / len(loader)
    avg_trans = total_trans_loss / len(loader)
    
    return avg_loss, avg_pm, avg_geo, avg_trans

def validate(model, loader, device, args, pm_loss_fn, l1_loss_fn):
    model.eval()
    total_loss = 0
    total_pm_loss = 0
    total_geo_loss = 0
    total_trans_loss = 0
    
    val_geo_error_sum = 0.0
    val_trans_error_sum = 0.0
    val_count = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            pts = batch['points'].to(device)
            gt_R = batch['gt_rot'].to(device)
            gt_t = batch['gt_t_residual'].to(device)
            obj_id = batch['obj_id'].to(device)
            sym = batch['sym_str']
            valid_points = batch['valid_points']
            xyz = pts[:,:,:3]

            # Filter out invalid samples
            mask = valid_points >= 50
            if not mask.any():
                continue
            
            pts = pts[mask]
            gt_R = gt_R[mask]
            gt_t = gt_t[mask]
            obj_id = obj_id[mask]
            xyz = xyz[mask]
            sym = [s for i, s in enumerate(sym) if mask[i]]
            
            pred_rot6d, pred_t = model(pts, obj_id)
            pred_R = utils.rotation_6d_to_matrix(pred_rot6d)

            loss_pm = pm_loss_fn(pred_R, pred_t, gt_R, gt_t, xyz, sym)
            loss_t = l1_loss_fn(pred_t, gt_t)
            loss_geo = loss_utils.symmetry_aware_geodesic_loss(pred_R, gt_R, sym)
            
            loss = args.w_pm * loss_pm + args.w_geo * loss_geo + args.w_trans * loss_t
            
            # Weighted by batch size for accurate average
            batch_size = pts.shape[0]
            total_loss += loss.item() * batch_size
            total_pm_loss += loss_pm.item() * batch_size
            total_geo_loss += loss_geo.item() * batch_size
            total_trans_loss += loss_t.item() * batch_size

            # Metrics
            val_geo_error_sum += loss_geo.item() * batch_size # Geo loss is in radians, convert later
            trans_dist = torch.norm(pred_t - gt_t, dim=1) * 100.0 # cm
            val_trans_error_sum += trans_dist.sum().item()
            
            val_count += batch_size

    if val_count == 0:
        return 0, 0, 0, 0, 0, 0

    avg_loss = total_loss / val_count
    avg_pm = total_pm_loss / val_count
    avg_geo = total_geo_loss / val_count
    avg_trans = total_trans_loss / val_count
    
    avg_rot_error_deg = (val_geo_error_sum / val_count) * (180.0 / np.pi)
    avg_trans_error_cm = val_trans_error_sum / val_count
    
    return avg_loss, avg_pm, avg_geo, avg_trans, avg_rot_error_deg, avg_trans_error_cm

def train():
    args = config.get_config()
    
    print(f"Total Epochs: {args.epochs}")
    
    # Save config
    os.makedirs(args.output_dir, exist_ok=True)
        
    DEVICE = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. Setup Datasets
    train_dataset = PoseDataset("train", args.training_data_dir, args.split_dir, num_points=args.num_points, subset_size=None)
    
    print("Calculating class weights for balanced sampling...")
    train_targets = [s[1] for s in train_dataset.samples]
    class_counts = Counter(train_targets)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = torch.DoubleTensor([class_weights[t] for t in train_targets])
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    print("WeightedRandomSampler initialized.")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    val_dataset = PoseDataset("val", args.training_data_dir, args.split_dir, num_points=args.num_points, subset_size=None)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    # 2. Setup Model
    print("Using PointNet")
    model = PointNet(num_classes=args.num_classes).to(DEVICE)
    
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    pm_loss_fn = loss_utils.point_matching_loss
    l1 = nn.L1Loss()
    writer = SummaryWriter()

    # Save config to run directory
    with open(os.path.join(writer.log_dir, 'config.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

    best_val_loss = float('inf')
    save_path = args.checkpoint_path
    
    # Ensure checkpoint directory exists
    checkpoint_dir = os.path.dirname(save_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        # Train
        t_loss, t_pm, t_geo, t_trans = train_one_epoch(model, train_loader, opt, DEVICE, args, pm_loss_fn, l1)
        
        writer.add_scalar('Train/Loss', t_loss, epoch)
        writer.add_scalar('Train/PM_Loss', t_pm, epoch)
        writer.add_scalar('Train/Geo_Loss', t_geo, epoch)
        writer.add_scalar('Train/Trans_Loss', t_trans, epoch)
        
        scheduler.step()
        
        # Validate
        v_loss, v_pm, v_geo, v_trans, v_rot_err, v_trans_err = validate(model, val_loader, DEVICE, args, pm_loss_fn, l1)
        
        writer.add_scalar('Val/Loss', v_loss, epoch)
        writer.add_scalar('Val/PM_Loss', v_pm, epoch)
        writer.add_scalar('Val/Geo_Loss', v_geo, epoch)
        writer.add_scalar('Val/Trans_Loss', v_trans, epoch)
        writer.add_scalar('Val/RotError', v_rot_err, epoch)
        writer.add_scalar('Val/TransError', v_trans_err, epoch)

        print(f"Epoch {epoch+1} | Train Loss {t_loss:.4f} | Val Loss {v_loss:.4f} | Val Rot {v_rot_err:.2f} deg | Val Trans {v_trans_err:.2f} cm")

        # Save Best
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss
            }, save_path)
            print(f"Saved best model to {save_path}")

if __name__ == "__main__":
    train()

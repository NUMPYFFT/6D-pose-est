import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
# ``tqdm.notebook`` can be left half-initialized when VS Code interrupts a cell.
# The standard renderer is stable in VS Code's notebook output and terminals.
from tqdm import tqdm
from collections import Counter
import cv2
import helpers as utils
import model as loss_utils
from helpers import PoseDataset
from model import build_pose_model
import helpers as config


def predict_pose(model, batch, points, obj_id, device, model_name):
    """Run either the geometry-only baseline or RGB--point fusion model."""
    if model_name == "fusion":
        image = batch['image'].to(device, non_blocking=True)
        pixel_coords = batch['pixel_coords'].to(device, non_blocking=True)
        rotation, translation = model(points, obj_id, image, pixel_coords)
        return rotation, translation, None
    if model_name == "correspondence":
        return model(
            points, obj_id,
            batch["source_points"].to(device, non_blocking=True),
        )
    if model_name == "multihypothesis":
        return model(points, obj_id)
    if model_name == "confidence":
        return model(points, obj_id)
    rotation, translation = model(points, obj_id)
    return rotation, translation, None


def multi_hypothesis_loss(auxiliary, gt_R, gt_t, sym, args):
    """Winner-takes-best pose loss plus confidence and light diversity terms."""
    hypothesis_rot6d = auxiliary["hypothesis_rot6d"]
    hypothesis_t = auxiliary["hypothesis_translation"]
    batch_size, hypothesis_count, _ = hypothesis_rot6d.shape
    matrices = torch.stack([
        utils.rotation_6d_to_matrix(hypothesis_rot6d[:, index])
        for index in range(hypothesis_count)
    ], dim=1)
    geo = torch.stack([
        loss_utils.symmetry_aware_geodesic_per_sample(matrices[:, index], gt_R, sym)
        for index in range(hypothesis_count)
    ], dim=1)
    translation = (hypothesis_t - gt_t.unsqueeze(1)).abs().mean(dim=2)
    candidate_error = geo + translation
    winners = candidate_error.argmin(dim=1)
    winner_geo = geo.gather(1, winners.unsqueeze(1)).mean()
    winner_translation = translation.gather(1, winners.unsqueeze(1)).mean()
    confidence_loss = F.cross_entropy(auxiliary["confidence_logits"], winners)

    # Avoid all four heads collapsing to the same rotation. A modest 20-degree
    # separation leaves room for equivalent symmetric poses.
    diversity_terms = []
    for left in range(hypothesis_count):
        for right in range(left + 1, hypothesis_count):
            relative = matrices[:, left].transpose(1, 2) @ matrices[:, right]
            trace = relative[:, 0, 0] + relative[:, 1, 1] + relative[:, 2, 2]
            angle = torch.acos(torch.clamp((trace - 1.0) * 0.5, -0.999, 0.999))
            diversity_terms.append(F.relu(0.35 - angle).mean())
    diversity_loss = torch.stack(diversity_terms).mean()
    loss = winner_geo + winner_translation + args.w_conf * confidence_loss + args.w_diversity * diversity_loss
    return loss, torch.zeros_like(loss), winner_geo, winner_translation

def train_one_epoch(model, loader, opt, device, args, pm_loss_fn, l1_loss_fn):
    model.train()
    total_loss = 0
    total_pm_loss = 0
    total_geo_loss = 0
    total_trans_loss = 0
    
    train_bar = tqdm(loader, desc="Train batches", unit="batch", leave=False, dynamic_ncols=True)
    for batch in train_bar:
        pts = batch['points'].to(device, non_blocking=True)
        gt_R = batch['gt_rot'].to(device, non_blocking=True)
        gt_t = batch['gt_t_residual'].to(device, non_blocking=True)
        obj_id = batch['obj_id'].to(device, non_blocking=True)
        sym = batch['sym_str']
        xyz = pts[:,:,:3]

        opt.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            pred_rot6d, pred_t, auxiliary = predict_pose(model, batch, pts, obj_id, device, args.model)
            pred_R = utils.rotation_6d_to_matrix(pred_rot6d)

            if auxiliary is not None and "hypothesis_rot6d" in auxiliary:
                loss, loss_pm, loss_geo, loss_t = multi_hypothesis_loss(
                    auxiliary, gt_R, gt_t, sym, args
                )
            else:
                loss_pm = pm_loss_fn(pred_R, pred_t, gt_R, gt_t, xyz, sym)
                loss_t = l1_loss_fn(pred_t, gt_t)
                loss_geo = loss_utils.symmetry_aware_geodesic_loss(pred_R, gt_R, sym)
                loss = args.w_pm * loss_pm + args.w_geo * loss_geo + args.w_trans * loss_t
            if auxiliary is not None and "matched_source" in auxiliary:
                # Source-point identity is ambiguous for symmetric objects, so
                # supervise soft source matches only on explicitly asymmetric
                # objects. For those objects, inverse-transforming each observed
                # point gives its canonical-coordinate target.
                asymmetric = torch.tensor(
                    [str(value).strip().lower() == "no" for value in sym],
                    device=device, dtype=torch.bool,
                )
                if asymmetric.any():
                    canonical_target = torch.bmm(
                        (xyz - gt_t.unsqueeze(1)), gt_R
                    )
                    corr_loss = F.smooth_l1_loss(
                        auxiliary["matched_source"][asymmetric],
                        canonical_target[asymmetric],
                    )
                    loss = loss + args.w_corr * corr_loss
        
        loss.backward()
        opt.step()
        
        total_loss += loss.item()
        total_pm_loss += loss_pm.item()
        total_geo_loss += loss_geo.item()
        total_trans_loss += loss_t.item()
        train_bar.set_postfix(loss=f"{loss.item():.4f}")

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
        validation_bar = tqdm(loader, desc="Validation batches", unit="batch", leave=False, dynamic_ncols=True)
        for batch in validation_bar:
            pts = batch['points'].to(device, non_blocking=True)
            gt_R = batch['gt_rot'].to(device, non_blocking=True)
            gt_t = batch['gt_t_residual'].to(device, non_blocking=True)
            obj_id = batch['obj_id'].to(device, non_blocking=True)
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
            # Fusion inputs are per-instance tensors too. Keep them aligned with
            # the filtered point-cloud batch before sampling image features.
            if args.model == "fusion":
                batch["image"] = batch["image"][mask]
                batch["pixel_coords"] = batch["pixel_coords"][mask]
            if args.model == "correspondence":
                batch["source_points"] = batch["source_points"][mask]
            
            pred_rot6d, pred_t, auxiliary = predict_pose(model, batch, pts, obj_id, device, args.model)
            pred_R = utils.rotation_6d_to_matrix(pred_rot6d)

            if auxiliary is not None and "hypothesis_rot6d" in auxiliary:
                loss, loss_pm, loss_geo, loss_t = multi_hypothesis_loss(
                    auxiliary, gt_R, gt_t, sym, args
                )
            else:
                loss_pm = pm_loss_fn(pred_R, pred_t, gt_R, gt_t, xyz, sym)
                loss_t = l1_loss_fn(pred_t, gt_t)
                loss_geo = loss_utils.symmetry_aware_geodesic_loss(pred_R, gt_R, sym)
                loss = args.w_pm * loss_pm + args.w_geo * loss_geo + args.w_trans * loss_t
            if auxiliary is not None and "matched_source" in auxiliary:
                asymmetric = torch.tensor(
                    [str(value).strip().lower() == "no" for value in sym],
                    device=device, dtype=torch.bool,
                )
                if asymmetric.any():
                    canonical_target = torch.bmm(
                        (xyz - gt_t.unsqueeze(1)), gt_R
                    )
                    corr_loss = F.smooth_l1_loss(
                        auxiliary["matched_source"][asymmetric],
                        canonical_target[asymmetric],
                    )
                    loss = loss + args.w_corr * corr_loss
            
            # Weighted by batch size for accurate average
            batch_size = pts.shape[0]
            total_loss += loss.item() * batch_size
            total_pm_loss += loss_pm.item() * batch_size
            total_geo_loss += loss_geo.item() * batch_size
            total_trans_loss += loss_t.item() * batch_size

            # For multi-hypothesis training, loss_geo is the oracle best-candidate
            # value. Report the confidence-selected hypothesis instead, matching
            # the pose that inference will actually send to ICP.
            metric_geo = (
                loss_utils.symmetry_aware_geodesic_loss(pred_R, gt_R, sym)
                if auxiliary is not None and "hypothesis_rot6d" in auxiliary
                else loss_geo
            )
            val_geo_error_sum += metric_geo.item() * batch_size # radians, converted later
            trans_dist = torch.norm(pred_t - gt_t, dim=1) * 100.0 # cm
            val_trans_error_sum += trans_dist.sum().item()
            
            val_count += batch_size
            validation_bar.set_postfix(loss=f"{loss.item():.4f}")

    if val_count == 0:
        return 0, 0, 0, 0, 0, 0

    avg_loss = total_loss / val_count
    avg_pm = total_pm_loss / val_count
    avg_geo = total_geo_loss / val_count
    avg_trans = total_trans_loss / val_count
    
    avg_rot_error_deg = (val_geo_error_sum / val_count) * (180.0 / np.pi)
    avg_trans_error_cm = val_trans_error_sum / val_count
    
    return avg_loss, avg_pm, avg_geo, avg_trans, avg_rot_error_deg, avg_trans_error_cm


def checkpoint_payload(model, optimizer, scheduler, epoch_idx, best_val_loss, args,
                       val_loss=None, val_rot_error=None, val_trans_error=None):
    """Create a self-contained checkpoint for best-model and periodic snapshots."""
    return {
        'epoch': epoch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'val_loss': val_loss,
        'val_rot_error_deg': val_rot_error,
        'val_trans_error_cm': val_trans_error,
        'fine_tune_from': args.init_checkpoint,
        'fine_tune_epochs_completed': epoch_idx + 1 if args.init_checkpoint else None,
    }

def train():
    args = config.get_config()
    
    print(f"Total Epochs: {args.epochs}")
    
    # Save config
    os.makedirs(args.output_dir, exist_ok=True)
        
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Install a CUDA-enabled PyTorch build "
            "or run explicitly with --device cpu. Training is stopped to avoid an "
            "unexpected CPU-only run."
        )
    DEVICE = torch.device(args.device)
    print(f"Training device: {DEVICE}")
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # 1. Setup Datasets
    uses_image = args.model == "fusion"
    uses_source = args.model == "correspondence"
    train_dataset = PoseDataset(
        "train", args.training_data_dir, args.split_dir, num_points=args.num_points,
        subset_size=None, return_image=uses_image, image_size=args.image_size,
        return_source=uses_source, num_source_points=args.num_source_points,
    )
    
    print("Calculating class weights for balanced sampling...")
    train_targets = [s[1] for s in train_dataset.samples]
    class_counts = Counter(train_targets)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = torch.DoubleTensor([class_weights[t] for t in train_targets])
    if args.hard_indices:
        if not os.path.isfile(args.hard_indices):
            raise FileNotFoundError(f"Hard-example index file was not found: {args.hard_indices}")
        hard_indices = np.load(args.hard_indices)
        hard_indices = np.asarray(hard_indices, dtype=np.int64)
        hard_indices = hard_indices[(hard_indices >= 0) & (hard_indices < len(train_dataset))]
        if len(hard_indices) == 0:
            raise ValueError("The hard-example index file contains no valid training indices.")
        sample_weights[torch.from_numpy(np.unique(hard_indices))] *= args.hard_replay_factor
        print(
            f"Hard-example replay: {len(np.unique(hard_indices))} instances weighted "
            f"{args.hard_replay_factor:.1f}x."
        )
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    print("WeightedRandomSampler initialized.")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    val_dataset = PoseDataset(
        "val", args.training_data_dir, args.split_dir, num_points=args.num_points,
        subset_size=None, return_image=uses_image, image_size=args.image_size,
        return_source=uses_source, num_source_points=args.num_source_points,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    # 2. Setup Model
    print(f"Using Model: {args.model}")
    model = build_pose_model(args.model, num_classes=args.num_classes).to(DEVICE)

    initial_checkpoint = None
    if args.init_checkpoint:
        if not os.path.isfile(args.init_checkpoint):
            raise FileNotFoundError(f"Initial checkpoint was not found: {args.init_checkpoint}")
        if os.path.abspath(args.init_checkpoint) == os.path.abspath(args.checkpoint_path):
            raise ValueError("--init_checkpoint and --checkpoint_path must be different files.")
        initial_checkpoint = torch.load(args.init_checkpoint, map_location=DEVICE)
        model.load_state_dict(initial_checkpoint["model_state_dict"])
        print(f"Fine-tuning from: {args.init_checkpoint}")
    
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    pm_loss_fn = loss_utils.point_matching_loss
    l1 = nn.L1Loss()
    writer = SummaryWriter()

    # Save config to run directory
    with open(os.path.join(writer.log_dir, 'config.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

    # For fine-tuning, the source model is already the initial best candidate.
    # Saving it to the new path ensures that the output remains a valid checkpoint
    # even when no later epoch improves validation loss.
    best_val_loss = float('inf')
    save_path = args.checkpoint_path
    
    # Ensure checkpoint directory exists
    checkpoint_dir = os.path.dirname(save_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    if initial_checkpoint is not None:
        source_best_val_loss = initial_checkpoint.get('best_val_loss', float('inf'))
        best_val_loss = float('inf') if args.reset_best_val else source_best_val_loss
        torch.save(checkpoint_payload(
            model, opt, scheduler, initial_checkpoint.get('epoch', -1),
            best_val_loss, args,
        ), save_path)
        if args.reset_best_val:
            print(f"Copied initial checkpoint to {save_path}; selecting the best validation result from this fine-tuning run.")
        else:
            print(f"Copied initial best checkpoint to {save_path} (Val Loss {best_val_loss:.4f})")

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Training epochs", unit="epoch", dynamic_ncols=True)
    for epoch in epoch_bar:
        epoch_idx = epoch - 1
        # Train
        t_loss, t_pm, t_geo, t_trans = train_one_epoch(model, train_loader, opt, DEVICE, args, pm_loss_fn, l1)
        
        writer.add_scalar('Train/Loss', t_loss, epoch_idx)
        writer.add_scalar('Train/PM_Loss', t_pm, epoch_idx)
        writer.add_scalar('Train/Geo_Loss', t_geo, epoch_idx)
        writer.add_scalar('Train/Trans_Loss', t_trans, epoch_idx)
        
        scheduler.step()
        
        # Validate
        v_loss, v_pm, v_geo, v_trans, v_rot_err, v_trans_err = validate(model, val_loader, DEVICE, args, pm_loss_fn, l1)
        
        writer.add_scalar('Val/Loss', v_loss, epoch_idx)
        writer.add_scalar('Val/PM_Loss', v_pm, epoch_idx)
        writer.add_scalar('Val/Geo_Loss', v_geo, epoch_idx)
        writer.add_scalar('Val/Trans_Loss', v_trans, epoch_idx)
        writer.add_scalar('Val/RotError', v_rot_err, epoch_idx)
        writer.add_scalar('Val/TransError', v_trans_err, epoch_idx)

        epoch_bar.set_postfix(
            train_loss=f"{t_loss:.4f}",
            val_loss=f"{v_loss:.4f}",
            rot_deg=f"{v_rot_err:.2f}",
            trans_cm=f"{v_trans_err:.2f}",
        )

        if args.snapshot_every > 0 and epoch % args.snapshot_every == 0:
            snapshot_dir = os.path.join(
                checkpoint_dir or ".", "snapshots",
                os.path.splitext(os.path.basename(save_path))[0],
            )
            os.makedirs(snapshot_dir, exist_ok=True)
            snapshot_path = os.path.join(snapshot_dir, f"epoch_{epoch:04d}.pth")
            torch.save(checkpoint_payload(
                model, opt, scheduler, epoch_idx, best_val_loss, args,
                val_loss=v_loss, val_rot_error=v_rot_err, val_trans_error=v_trans_err,
            ), snapshot_path)
            print(f"Saved periodic snapshot to {snapshot_path}")

        # Save Best
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(checkpoint_payload(
                model, opt, scheduler, epoch_idx, best_val_loss, args,
                val_loss=v_loss, val_rot_error=v_rot_err, val_trans_error=v_trans_err,
            ), save_path)

    print(f"Training complete | best validation loss: {best_val_loss:.4f} | checkpoint: {save_path}")

if __name__ == "__main__":
    train()

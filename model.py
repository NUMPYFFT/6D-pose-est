import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNet(nn.Module):
    def __init__(self, num_classes=79):
        super().__init__()
        self.conv1 = nn.Conv1d(9, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.obj_emb = nn.Embedding(num_classes + 1, 128)
        self.rot_head = nn.Sequential(
            nn.Linear(1152, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 6),
        )
        self.trans_head = nn.Sequential(
            nn.Linear(1152, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 3),
        )

    def forward(self, x, obj_id):
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0].view(-1, 1024)
        x = torch.cat([x, self.obj_emb(obj_id)], dim=1)
        return self.rot_head(x), self.trans_head(x)


class PointNetBaseline(PointNet):
    """Named entry point for the checkpoint-compatible PointNet baseline."""
    pass


class PointImageFusion(nn.Module):
    """Fuse masked RGB appearance with observed-point geometry for 6D pose.

    The dataset supplies a masked RGB crop and the projected crop coordinate of
    every observed point.  ``grid_sample`` then gathers a CNN feature at each
    point, preserving the direct image-to-geometry correspondence shown in the
    design diagram.  This is deliberately a compact first fusion model; the
    existing PointNet remains available as an unchanged baseline.
    """

    def __init__(self, num_classes=79, image_channels=4):
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Conv1d(9, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.image_encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(256 + 128 + 1, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.obj_emb = nn.Embedding(num_classes + 1, 128)
        head_dim = 256 + 256 + 128
        self.rot_head = nn.Sequential(
            nn.Linear(head_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 6),
        )
        self.trans_head = nn.Sequential(
            nn.Linear(head_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 3),
        )

    def forward(self, points, obj_id, image, pixel_coords):
        """Predict pose from points, a masked image crop, and normalized UVs.

        ``pixel_coords`` has shape ``[B, N, 2]`` in the ``[-1, 1]`` coordinate
        system required by :func:`torch.nn.functional.grid_sample`.
        """
        point_features = self.point_encoder(points.transpose(2, 1))
        image_features = self.image_encoder(image)
        sample_grid = pixel_coords.unsqueeze(2)
        image_at_points = F.grid_sample(
            image_features, sample_grid, mode="bilinear", padding_mode="zeros",
            align_corners=True,
        ).squeeze(-1)
        in_crop = ((pixel_coords.abs() <= 1).all(dim=-1, keepdim=True)
                   .transpose(2, 1).to(points.dtype))
        fused = self.fusion(torch.cat([point_features, image_at_points, in_crop], dim=1))
        global_fused = fused.max(dim=2).values
        global_geometry = point_features.max(dim=2).values
        global_features = torch.cat(
            [global_fused, global_geometry, self.obj_emb(obj_id)], dim=1
        )
        return self.rot_head(global_features), self.trans_head(global_features)


def build_pose_model(model_name, num_classes=79):
    """Create one of the two retained pose initializers."""
    if model_name == "pointnet":
        return PointNetBaseline(num_classes=num_classes)
    if model_name == "fusion":
        return PointImageFusion(num_classes=num_classes)
    raise ValueError(f"Unknown pose model: {model_name}")

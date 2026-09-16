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

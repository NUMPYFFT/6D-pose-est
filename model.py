import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self, num_classes=79):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(9, 64, 1) # Input channels: 9 (XYZ + RGB + Normals)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # Embedding for Object ID
        self.obj_emb = nn.Embedding(num_classes + 1, 128) # +1 for safety

        # Decoupled Heads with Dropout
        self.rot_head = nn.Sequential(
            nn.Linear(1024 + 128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 6)
        )

        self.trans_head = nn.Sequential(
            nn.Linear(1024 + 128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)
        )

    def forward(self, x, obj_id):
        # x: (B, N, 9) -> (B, 9, N)
        x = x.transpose(2, 1)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # Object Embedding
        emb = self.obj_emb(obj_id) # (B, 128)
        
        # Concatenate
        x = torch.cat([x, emb], dim=1) # (B, 1152)

        rot = self.rot_head(x)
        trans = self.trans_head(x)
        
        return rot, trans

import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature

class DGCNN(nn.Module):
    def __init__(self, num_classes=79, k=20):
        super(DGCNN, self).__init__()
        self.k = k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        # Input is 9 channels (XYZ+RGB+N), so EdgeConv input is 9*2 = 18
        self.conv1 = nn.Sequential(nn.Conv2d(18, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        # Aggregation: 64+64+128+256 = 512
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        # Embedding for Object ID
        self.obj_emb = nn.Embedding(num_classes + 1, 128)

        # Heads
        self.rot_head = nn.Sequential(
            nn.Linear(1024 + 128, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.5),
            nn.Linear(256, 6)
        )

        self.trans_head = nn.Sequential(
            nn.Linear(1024 + 128, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.5),
            nn.Linear(256, 3)
        )

    def forward(self, x, obj_id):
        # x: (B, N, 9) -> (B, 9, N)
        x = x.transpose(2, 1)
        batch_size = x.size(0)

        x1 = get_graph_feature(x, k=self.k)     # (B, 9, N) -> (B, 18, N, k)
        x1 = self.conv1(x1)                     # (B, 64, N, k)
        x1 = x1.max(dim=-1, keepdim=False)[0]   # (B, 64, N)

        x2 = get_graph_feature(x1, k=self.k)    # (B, 64, N) -> (B, 128, N, k)
        x2 = self.conv2(x2)                     # (B, 64, N, k)
        x2 = x2.max(dim=-1, keepdim=False)[0]   # (B, 64, N)

        x3 = get_graph_feature(x2, k=self.k)    # (B, 64, N) -> (B, 128, N, k)
        x3 = self.conv3(x3)                     # (B, 128, N, k)
        x3 = x3.max(dim=-1, keepdim=False)[0]   # (B, 128, N)

        x4 = get_graph_feature(x3, k=self.k)    # (B, 128, N) -> (B, 256, N, k)
        x4 = self.conv4(x4)                     # (B, 256, N, k)
        x4 = x4.max(dim=-1, keepdim=False)[0]   # (B, 256, N)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (B, 512, N)

        x = self.conv5(x)                       # (B, 1024, N)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) # (B, 1024)

        # Object Embedding
        emb = self.obj_emb(obj_id) # (B, 128)
        
        # Concatenate
        x = torch.cat([x, emb], dim=1) # (B, 1152)

        rot = self.rot_head(x)
        trans = self.trans_head(x)
        
        return rot, trans

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

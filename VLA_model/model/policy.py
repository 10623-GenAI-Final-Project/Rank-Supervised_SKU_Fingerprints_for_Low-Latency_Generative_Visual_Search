import torch
import torch.nn as nn
import torch.nn.functional as F

class VLAPolicy(nn.Module):
    def __init__(self, visual_dim, quality_dim, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(visual_dim + quality_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, num_actions)

    def forward(self, vfeat, qfeat):
        x = torch.cat([vfeat, qfeat], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
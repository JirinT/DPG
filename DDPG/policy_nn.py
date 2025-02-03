import torch
from torch import nn

class DeterministicPolicy(nn.Module):
    def __init__(self, state_space):
        super().__init__()
        self.linear_net = nn.Sequential(
            nn.Linear(state_space, 64),
            nn.ReLU(),
            # nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            # nn.BatchNorm1d(32),
            nn.Linear(32,1)          
        )

    def forward(self, x):
        action = self.linear_net(x)
        return torch.tanh(acti
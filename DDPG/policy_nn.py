import torch
from torch import nn

class DeterministicPolicy(nn.Module):
    def __init__(self, state_space, learning_rate):
        super().__init__()
        self.linear_net = nn.Sequential(
            nn.Linear(state_space, 400),
            nn.ReLU(),
            # nn.BatchNorm1d(64),
            nn.Linear(400, 300),
            nn.ReLU(),
            # nn.BatchNorm1d(32),
            nn.Linear(300,1)          
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        action = self.linear_net(x)
        return torch.tanh(action)

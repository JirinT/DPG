import torch
from torch import nn
from adabound import AdaBound

class DeterministicPolicy(nn.Module):
    def __init__(self, state_space, learning_rate):
        super().__init__()
        self.linear_net = nn.Sequential(
            nn.Linear(state_space, 400),
            nn.ReLU(),
            # nn.BatchNorm1d(400),
            nn.Linear(400, 300),
            nn.ReLU(),
            # nn.BatchNorm1d(300),
            nn.Linear(300, 1)          
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        # self.optimizer = AdaBound(self.parameters(), lr=0.0001, final_lr=0.01)

    def forward(self, x):
        action = self.linear_net(x)

        return torch.tanh(action)

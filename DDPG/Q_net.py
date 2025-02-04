import torch
from torch import nn

class Q_nn(nn.Module):
    def __init__(self, state_space, action_space, learning_rate):
        super().__init__()
        input_features = state_space + action_space
        self.linear_sequence = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, action_space)
        )

        self.loss_fcn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        Q_value = self.linear_sequence(x)
        return Q_value
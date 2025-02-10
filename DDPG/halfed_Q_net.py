import torch
import torch.nn as nn
import torch.functional as F

class Q_nn(nn.Module):
    def __init__(self, state_space, action_space, learning_rate):
        super().__init__()
        self.fc1 = nn.Linear(state_space, 400)
        self.fc2 = nn.Linear(400+action_space, 300)
        self.fc3 = nn.Linear(300, 1)
        self.relu = nn.ReLU()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
    def forward(self, states, action):
        fc1 = self.fc1(states)
        fc1_relu = self.relu(fc1)
        fc2 = self.fc2(torch.cat((fc1_relu, action), dim=1))
        fc2_relu = self.relu(fc2)
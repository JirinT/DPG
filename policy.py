import torch

# linear aproximation of policy:
class Policy:
    def __init__(self, parameters):
        self.theta = parameters

    def __call__(self, state):
        action = torch.dot(self.theta, state)
        action = torch.tanh(action) # make sure the action is in (-1,1) range

        return action
import torch

class ValueFunction:
    def __init__(self, parameters):
        self.v = parameters # shape 1xstate_space

    def __call__(self, state):
        return torch.dot(self.v, state) # linear approximation
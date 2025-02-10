import torch

class Normalizer():
    def __init__(self, num_features, momentum=0.9):
        self.mean = torch.zeros(num_features)
        self.std = torch.ones(num_features)
        self.momentum = momentum
        self.steps = 0
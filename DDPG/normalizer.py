import torch

class Normalizer():
    def __init__(self, num_features, momentum=0.9):
        self.mean = torch.zeros(num_features)
        self.std = torch.ones(num_features)
        self.momentum = momentum
        self.steps = 0
    
    def normalize(self, in_features):
        # compute new mean&std:
        curr_std, curr_mean = torch.std_mean(in_features, dim=0)
        # update running mean&std:
        self.mean = self.momentum*self.mean + (1-self.momentum)*curr_mean
        self.std = self.momentum*self.std + (1-self.momentum)*curr_std

        # normalize the input with the running mean&std:
        output = in_features - self.mean / (torch.sqrt(self.std) + 1e-8)

        return output
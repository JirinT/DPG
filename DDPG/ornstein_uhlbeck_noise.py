import torch
import numpy as np

class OU_noise():
    def __init__(self, theta, sigma, dt=0.01):
        self.noise = 0
        self.theta = theta
        self.sigma = sigma
        self.dt = dt

    def sample(self):
        # self.noise += self.theta*self.noise*self.dt + self.sigma*torch.randn(1)
        self.noise += -self.theta*self.noise*self.dt + self.sigma*np.sqrt(self.dt)*np.random.normal(loc=0, scale=1)
        return torch.tensor(self.noise)
    
    def reset(self):
        self.noise = 0
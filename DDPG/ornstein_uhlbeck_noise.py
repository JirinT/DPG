import torch

class OU_noise():
    def __init__(self, lam, sigma):
        self.noise = 0
        self.lambda_koef = lam
        self.sigma = sigma
    
    def sample(self):
        self.noise = self.lambda_koef*self.noise + self.sigma*torch.randn(1)
        return self.noise
    
    def reset(self):
        self.noise = 0
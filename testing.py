from DDPG.normalizer import Normalizer
import torch

normalizer = Normalizer(4)

features = torch.tensor([[1,1,1,1],
                         [2,2,2,2],
                         [3,3,3,3]], dtype=torch.float32)

print(normalizer.normalize(features))
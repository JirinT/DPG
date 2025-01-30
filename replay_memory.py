from collections import deque
import random

class ReplayMemory:
    def __init__(self, max_num_samples):
        self.memory = deque(maxlen=max_num_samples)
    
    def __len__(self):
        return len(self.memory)

    def push(self, experience):
        self.memory.append(experience)

    def sample_batch(self, amount):
        batch = random.sample(self.memory, amount)
        return batch
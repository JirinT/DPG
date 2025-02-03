from collections import deque
import random

class ReplayMemory():
    def __init__(self, mem_size):
        self.memory = deque(maxlen=mem_size)
    
    def __len__(self):
        return len(self.memory)
    
    def push(self, experience):
        self.memory.append(experience)

    def sample(self, amount):
        return random.sample(self.memory, amount)
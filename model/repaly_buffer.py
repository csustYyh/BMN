import collections
import random
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        if next_state is None:
            next_state = torch.zeros_like(state)
        self.buffer.append((state, action, reward, next_state, done))

#  train_batch_size: 4
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        state = torch.stack(state)
        action = torch.tensor(action)
        reward = torch.tensor(reward)
        next_state = torch.stack(next_state) 
        done = torch.tensor(done)
        return state.squeeze(dim=1), action, reward, next_state.squeeze(dim=1), done

    def size(self):
        return len(self.buffer)

    def remove(self):
        for i in range (1000):
            self.buffer.popleft()





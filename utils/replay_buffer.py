import random
from utils.configs import Config

class ReplayBuffer:
    def __init__(self, max_size=Config.REPLAY_BUFFER_SIZE):
        self.buffer = []
        self.max_size = max_size

    def push(self, state, action, reward, next_state, done, legal_next_actions):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done, legal_next_actions))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
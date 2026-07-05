import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def store_batch(self, states, actions, rewards, next_states, dones):
        """Store a batch collected from parallel environments."""
        for state, action, reward, next_state, done in zip(
            states,
            actions,
            rewards,
            next_states,
            dones,
        ):
            self.store(
                np.asarray(state, dtype=np.float32).copy(),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32).copy(),
                float(done),
            )
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32) # using float32 for dones because of how it's used in Bellman equation
        )
        
    def __len__(self):
        return len(self.buffer)

import numpy as np


class RolloutBuffer:
    def __init__(self):
        self.clear()
        
    def store(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
    def compute_gae(self, last_value, gamma=0.99, lam=0.95):
        advantages = []
        gae = 0
        values = self.values + [last_value]
        
        for t in reversed(range(len(self.rewards))):
            mask = 1 - self.dones[t]
            delta = self.rewards[t] + gamma * values[t + 1] * mask - values[t]
            gae = delta + gamma * lam * mask * gae
            advantages.insert(0, gae)
        
        returns = [a + v for a, v in zip(advantages, self.values)]
            
        return advantages, returns


class VectorRolloutBuffer:
    """PPO rollout storage with a separate GAE trajectory per environment."""

    def __init__(self):
        self.clear()

    def store(self, states, actions, rewards, dones, log_probs, values):
        self.states.append(np.asarray(states, dtype=np.float32).copy())
        self.actions.append(np.asarray(actions, dtype=np.int64).copy())
        self.rewards.append(np.asarray(rewards, dtype=np.float32).copy())
        self.dones.append(np.asarray(dones, dtype=np.float32).copy())
        self.log_probs.append(np.asarray(log_probs, dtype=np.float32).copy())
        self.values.append(np.asarray(values, dtype=np.float32).copy())

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def compute_gae(self, last_values, gamma=0.99, lam=0.95):
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        last_values = np.asarray(last_values, dtype=np.float32)
        advantages = np.zeros_like(rewards)
        gae = np.zeros(rewards.shape[1], dtype=np.float32)

        for step in reversed(range(rewards.shape[0])):
            next_values = last_values if step == rewards.shape[0] - 1 else values[step + 1]
            mask = 1.0 - dones[step]
            delta = rewards[step] + gamma * next_values * mask - values[step]
            gae = delta + gamma * lam * mask * gae
            advantages[step] = gae

        returns = advantages + values
        return advantages.reshape(-1), returns.reshape(-1)

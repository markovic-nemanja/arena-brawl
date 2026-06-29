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
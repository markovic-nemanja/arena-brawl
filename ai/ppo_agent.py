import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from ai.ppo_network import ActorCritic

class PPOAgent:
    def __init__(self, state_size=20, action_size=10, lr=3e-4, gamma=0.99,
                 gae_lambda=0.95, clip_epsilon=0.2, epochs=4, batch_size=64,
                 entropy_coef=0.005, value_coef=0.5):

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network = ActorCritic(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def select_action(self, state):
        """Training: SAMPLE from the policy (exploration). Returns (action, log_prob, value)."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.network(state_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def act(self, state):
        """Evaluation/play: GREEDY (argmax of the logits). Returns just the action int."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.network(state_tensor)
        return torch.argmax(logits, dim=-1).item()

    def update(self, buffer, last_value):
        advantages, returns = buffer.compute_gae(last_value, self.gamma, self.gae_lambda)

        states = torch.FloatTensor(np.array(buffer.states)).to(self.device)
        actions = torch.LongTensor(buffer.actions).to(self.device)
        old_log_probs = torch.FloatTensor(buffer.log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalize advantages
        n = len(states)

        total_policy_loss = total_value_loss = total_entropy = 0
        num_batches = 0

        for _ in range(self.epochs):
            idx = torch.randperm(n)

            for start in range(0, n, self.batch_size):
                b = idx[start:start + self.batch_size]

                logits, values = self.network(states[b])
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions[b])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs[b])
                surr1 = ratio * advantages[b]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[b]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values.squeeze(1), returns[b])
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_batches += 1

        return (total_policy_loss / num_batches, total_value_loss / num_batches, total_entropy / num_batches)

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        self.network.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))

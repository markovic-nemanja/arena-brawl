import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from ai.A2CNetwork import A2CNetwork


class A2CAgent:
    def __init__(self, state_size=22, action_size=10, lr=7e-4, gamma=0.99,
                 gae_lambda=0.95, entropy_coef=0.01, value_coef=0.5,
                 max_grad_norm=0.5):

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network = A2CNetwork(state_size, action_size).to(self.device)
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

    def select_actions(self, states):
        """Training: sample one action for every parallel environment."""
        state_tensor = torch.FloatTensor(states).to(self.device)
        with torch.no_grad():
            logits, values = self.network(state_tensor)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.squeeze(1).cpu().numpy(),
        )

    def get_values(self, states):
        """Return V(s) for every parallel environment."""
        state_tensor = torch.FloatTensor(states).to(self.device)
        with torch.no_grad():
            _, values = self.network(state_tensor)
        return values.squeeze(1).cpu().numpy()

    def act(self, state):
        """Evaluation/play: GREEDY (argmax of the logits). Returns just the action int."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.network(state_tensor)
        return torch.argmax(logits, dim=-1).item()

    def update(self, buffer, last_value):
        advantages, returns = buffer.compute_gae(last_value, self.gamma, self.gae_lambda)

        states_array = np.array(buffer.states)
        states = torch.FloatTensor(
            states_array.reshape(-1, states_array.shape[-1])
        ).to(self.device)
        actions = torch.LongTensor(np.array(buffer.actions).reshape(-1)).to(self.device)
        advantages = torch.FloatTensor(np.array(advantages).reshape(-1)).to(self.device)
        returns = torch.FloatTensor(np.array(returns).reshape(-1)).to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        logits, values = self.network(states)
        dist = Categorical(logits=logits)
        entropy = dist.entropy().mean()

        policy_loss = -(dist.log_prob(actions) * advantages.detach()).mean()
        value_loss = nn.functional.mse_loss(values.squeeze(1), returns)
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return policy_loss.item(), value_loss.item(), entropy.item()

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        self.network.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))

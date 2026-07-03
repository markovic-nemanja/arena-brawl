import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, state_size=34, action_size=10, hidden_size=128):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        self.actor = nn.Linear(hidden_size, action_size)  # Outputs logits over 10 actions
        self.critic = nn.Linear(hidden_size, 1)           # Outputs V(s) - state value estimate

    def forward(self, x):
        h = self.shared(x)
        logits = self.actor(h)
        value = self.critic(h)
        return logits, value

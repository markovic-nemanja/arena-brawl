import numpy as np
import torch
import torch.nn as nn


def _orthogonal_init(module, gain=np.sqrt(2)):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain)
        nn.init.constant_(module.bias, 0.0)
    return module


class ActorCritic(nn.Module):
    """Separate actor and critic trunks avoid value-learning interference."""

    def __init__(self, state_size=31, action_size=18, hidden_size=256):
        super().__init__()

        self.actor_body = nn.Sequential(
            _orthogonal_init(nn.Linear(state_size, hidden_size)),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )
        self.critic_body = nn.Sequential(
            _orthogonal_init(nn.Linear(state_size, hidden_size)),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )
        self.actor = _orthogonal_init(nn.Linear(hidden_size, action_size), gain=0.01)
        self.critic = _orthogonal_init(nn.Linear(hidden_size, 1), gain=1.0)

    def forward(self, x):
        logits = self.actor(self.actor_body(x))
        value = self.critic(self.critic_body(x))
        return logits, value

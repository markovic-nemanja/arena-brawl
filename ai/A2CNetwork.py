import math

import torch
import torch.nn as nn


class A2CNetwork(nn.Module):
    def __init__(
        self,
        state_size=22,
        action_size=10,
        hidden_size=128,
    ):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        self.actor = nn.Linear(hidden_size, action_size)

        self.critic = nn.Linear(hidden_size, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(
                    layer.weight,
                    gain=math.sqrt(2),
                )
                nn.init.zeros_(layer.bias)

        nn.init.orthogonal_(
            self.actor.weight,
            gain=0.01,
        )
        nn.init.zeros_(self.actor.bias)

        nn.init.orthogonal_(
            self.critic.weight,
            gain=1.0,
        )
        nn.init.zeros_(self.critic.bias)

    def forward(self, states):
        features = self.shared(states)

        action_logits = self.actor(features)
        state_values = self.critic(features)

        return action_logits, state_values
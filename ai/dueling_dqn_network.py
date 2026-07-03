import torch.nn as nn


class DuelingQNetwork(nn.Module):
    def __init__(
        self,
        state_size=34,
        action_size=10,
        hidden_size=128,
    ):
        super().__init__()

        self.feature_network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # V(s): procena vrednosti trenutnog stanja
        self.value_stream = nn.Linear(
            hidden_size,
            1,
        )

        # A(s,a): prednost svake akcije u trenutnom stanju
        self.advantage_stream = nn.Linear(
            hidden_size,
            action_size,
        )

    def forward(self, state):
        features = self.feature_network(state)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        q_values = (
            value
            + advantage
            - advantage.mean(dim=-1, keepdim=True)
        )

        return q_values

import random

import torch
import torch.nn as nn
import torch.optim as optim

from ai.dueling_dqn_network import DuelingQNetwork


class DuelingDQNAgent:
    def __init__(self, state_size=34, action_size=10,
                 learning_rate=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.999,
                 target_update_freq=1000, batch_size=64):

        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.step_count = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.online_network = DuelingQNetwork(state_size, action_size).to(self.device)
        self.target_network = DuelingQNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(
            self.online_network.parameters(),
            lr=learning_rate,
        )
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        """Select an epsilon-greedy action during training."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.online_network(state_tensor)

        return q_values.argmax(dim=1).item()

    def act(self, state):
        """Select a greedy action during evaluation or play."""
        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.online_network(state_tensor)

        return q_values.argmax(dim=1).item()

    def train_step(self, buffer):
        if len(buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = buffer.sample(
            self.batch_size
        )

        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(
            next_states,
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.online_network(states)
        selected_q_values = q_values.gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)

        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(dim=1).values
            targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = self.loss_fn(selected_q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.online_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay,
        )

    def save(self, path):
        torch.save(self.online_network.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(
            path,
            map_location=self.device,
            weights_only=True,
        )
        self.online_network.load_state_dict(state_dict)
        self.update_target_network()

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),   # first hidden layer
            nn.ReLU(),
            nn.Linear(128, 128),          # second hidden layer
            nn.ReLU(),
            nn.Linear(128, action_size)   # output layer (Q-values)
        )

    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_size=20, action_size=10,
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

        self.online_network = QNetwork(state_size, action_size).to(self.device)  # trained every step
        self.target_network = QNetwork(state_size, action_size).to(self.device)  # frozen copy for Bellman targets
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.online_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss() # Difference between network prediciton and Bellman target

    def select_action(self, state):
        """Training: epsilon-greedy (explore with prob epsilon, else argmax Q)."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        # state_tensor is array of shape (16,), unsqueeze(0) adds a new dimension at index 0, making it (1, 16), which is the expected input shape for the network
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_network(state_tensor)
        return q_values.argmax().item()

    def select_actions(self, states):
        """Epsilon-greedy action selection for parallel environments."""
        state_tensor = torch.as_tensor(
            states, dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            greedy_actions = self.online_network(state_tensor).argmax(dim=1)
        actions = greedy_actions.cpu().numpy()
        explore = np.random.random(len(actions)) < self.epsilon
        actions[explore] = np.random.randint(
            0, self.action_size, size=int(explore.sum())
        )
        return actions

    def act(self, state):
        """Evaluation/play: GREEDY (argmax Q, ignores epsilon). Returns just the action int."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_network(state_tensor)
        return q_values.argmax().item()

    def train_step(self, buffer):
        if len(buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.online_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(1)[0]

        targets = rewards + self.gamma * max_next_q * (1 - dones)  # Bellman equation

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad() # clears old gradients (otherwise they would accumulate)
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.online_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save(self.online_network.state_dict(), path)

    def load(self, path):
        self.online_network.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.update_target_network()

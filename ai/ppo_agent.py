import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from ai.ppo_network import ActorCritic
from systems.controller import RL_MOVEMENT_ACTIONS


class PPOAgent:
    def __init__(
        self,
        state_size=31,
        action_size=18,
        lr=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        epochs=4,
        batch_size=256,
        entropy_coef=0.01,
        final_entropy_coef=0.002,
        value_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.02,
        value_clip=0.2,
        device=None,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.initial_lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.initial_entropy_coef = entropy_coef
        self.final_entropy_coef = final_entropy_coef
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.value_clip = value_clip

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.network = ActorCritic(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

    def _masked_logits(self, logits, states):
        """Mask all fire actions while the normalized cooldown is non-zero."""
        if self.action_size <= RL_MOVEMENT_ACTIONS:
            return logits
        cooldown_active = states[..., 6] > 1e-6
        if not torch.any(cooldown_active):
            return logits
        valid = torch.ones_like(logits, dtype=torch.bool)
        valid[cooldown_active, RL_MOVEMENT_ACTIONS:] = False
        return logits.masked_fill(~valid, torch.finfo(logits.dtype).min)

    def select_action(self, state):
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.network(state_tensor)
            logits = self._masked_logits(logits, state_tensor)
        distribution = Categorical(logits=logits)
        action = distribution.sample()
        return (
            action.item(),
            distribution.log_prob(action).item(),
            value.item(),
        )

    def select_actions(self, states):
        state_tensor = torch.as_tensor(
            states, dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            logits, values = self.network(state_tensor)
            logits = self._masked_logits(logits, state_tensor)
        distribution = Categorical(logits=logits)
        actions = distribution.sample()
        return (
            actions.cpu().numpy(),
            distribution.log_prob(actions).cpu().numpy(),
            values.squeeze(1).cpu().numpy(),
        )

    def get_values(self, states):
        state_tensor = torch.as_tensor(
            states, dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            _, values = self.network(state_tensor)
        return values.squeeze(1).cpu().numpy()

    def act(self, state):
        """Deterministic masked action used for evaluation and playback."""
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.network(state_tensor)
            logits = self._masked_logits(logits, state_tensor)
        return torch.argmax(logits, dim=-1).item()

    def set_training_progress(self, progress):
        progress = float(np.clip(progress, 0.0, 1.0))
        # Retain 10% of the initial learning rate at the end instead of
        # completely freezing the policy during the final curriculum phase.
        learning_rate = self.initial_lr * (1.0 - 0.9 * progress)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        self.entropy_coef = (
            self.initial_entropy_coef * (1.0 - progress)
            + self.final_entropy_coef * progress
        )
        return learning_rate

    def update(self, buffer, last_values):
        advantages, returns = buffer.compute_gae(
            last_values, self.gamma, self.gae_lambda
        )

        state_array = np.asarray(buffer.states, dtype=np.float32)
        states = torch.as_tensor(
            state_array.reshape(-1, state_array.shape[-1]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            np.asarray(buffer.actions).reshape(-1),
            dtype=torch.long,
            device=self.device,
        )
        old_log_probs = torch.as_tensor(
            np.asarray(buffer.log_probs).reshape(-1),
            dtype=torch.float32,
            device=self.device,
        )
        old_values = torch.as_tensor(
            np.asarray(buffer.values).reshape(-1),
            dtype=torch.float32,
            device=self.device,
        )
        advantages = torch.as_tensor(
            advantages, dtype=torch.float32, device=self.device
        )
        returns = torch.as_tensor(
            returns, dtype=torch.float32, device=self.device
        )

        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
        n = len(states)
        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "grad_norm": 0.0,
        }
        num_batches = 0
        stopped_early = False

        for _ in range(self.epochs):
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                batch = indices[start:start + self.batch_size]
                logits, values = self.network(states[batch])
                logits = self._masked_logits(logits, states[batch])
                distribution = Categorical(logits=logits)
                new_log_probs = distribution.log_prob(actions[batch])
                entropy = distribution.entropy().mean()

                log_ratio = new_log_probs - old_log_probs[batch]
                ratio = torch.exp(log_ratio)
                surrogate_1 = ratio * advantages[batch]
                surrogate_2 = torch.clamp(
                    ratio,
                    1 - self.clip_epsilon,
                    1 + self.clip_epsilon,
                ) * advantages[batch]
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

                values = values.squeeze(1)
                clipped_values = old_values[batch] + torch.clamp(
                    values - old_values[batch],
                    -self.value_clip,
                    self.value_clip,
                )
                value_loss = torch.maximum(
                    (values - returns[batch]).pow(2),
                    (clipped_values - returns[batch]).pow(2),
                ).mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.clip_epsilon)
                        .float()
                        .mean()
                    )

                totals["policy_loss"] += policy_loss.item()
                totals["value_loss"] += value_loss.item()
                totals["entropy"] += entropy.item()
                totals["approx_kl"] += approx_kl.item()
                totals["clip_fraction"] += clip_fraction.item()
                totals["grad_norm"] += float(grad_norm)
                num_batches += 1

                if self.target_kl and approx_kl.item() > self.target_kl:
                    stopped_early = True
                    break
            if stopped_early:
                break

        for key in totals:
            totals[key] /= max(num_batches, 1)
        returns_np = returns.detach().cpu().numpy()
        values_np = old_values.detach().cpu().numpy()
        return_variance = np.var(returns_np)
        totals["explained_variance"] = (
            1.0 - np.var(returns_np - values_np) / return_variance
            if return_variance > 1e-8
            else 0.0
        )
        totals["early_stop"] = float(stopped_early)
        return totals

    def checkpoint(self, **extra):
        return {
            "format_version": 2,
            "algorithm": "ppo",
            "state_size": self.state_size,
            "action_size": self.action_size,
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "hyperparameters": {
                "lr": self.initial_lr,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_epsilon": self.clip_epsilon,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "entropy_coef": self.initial_entropy_coef,
                "final_entropy_coef": self.final_entropy_coef,
                "value_coef": self.value_coef,
                "max_grad_norm": self.max_grad_norm,
                "target_kl": self.target_kl,
                "value_clip": self.value_clip,
            },
            **extra,
        }

    def save(self, path, **extra):
        torch.save(self.checkpoint(**extra), path)

    def load(self, path, load_optimizer=False):
        checkpoint = torch.load(
            path, map_location=self.device, weights_only=True
        )
        if "network" in checkpoint:
            self.network.load_state_dict(checkpoint["network"])
            if load_optimizer and "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            return checkpoint

        # Legacy state-dict-only checkpoints remain recognizable, although
        # weights trained with the old 20x10 architecture cannot fit this model.
        self.network.load_state_dict(checkpoint)
        return {"format_version": 1}

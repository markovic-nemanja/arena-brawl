"""AIM TRAINING — isolate the aiming skill with parallel A2C rollouts.

The agent faces a StationaryBot that respawns at a NEW RANDOM position every time it's hit, and is
rewarded ONLY for damage dealt (no dodging, no winning, no nudges). Because the target keeps moving,
a fixed-direction policy scores nothing — the agent is forced to read the target's position and aim
at it. Parallel arenas collect one rollout together before each A2C update.

Run:  python -m ai.train_aim_a2c
"""
import os
import csv
import time
from functools import partial

import numpy as np
import wandb
from gymnasium.vector import AsyncVectorEnv, AutoresetMode

from arena_env import ArenaBrawlEnv
from systems.roles import Gunner
from systems.controller import StationaryBot
from ai.a2c_agent import A2CAgent


class ParallelRolloutBuffer:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.clear()

    def store(self, state, action, reward, done, log_prob, value):
        self.states.append(np.asarray(state, dtype=np.float32).copy())
        self.actions.append(np.asarray(action, dtype=np.int64).copy())
        self.rewards.append(np.asarray(reward, dtype=np.float32).copy())
        self.dones.append(np.asarray(done, dtype=np.float32).copy())
        self.log_probs.append(np.asarray(log_prob, dtype=np.float32).copy())
        self.values.append(np.asarray(value, dtype=np.float32).copy())

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def compute_gae(self, last_value, gamma=0.99, lam=0.95):
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = np.zeros(self.num_envs, dtype=np.float32)
        next_value = np.asarray(last_value, dtype=np.float32)

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * mask - values[t]
            gae = delta + gamma * lam * mask * gae
            advantages[t] = gae
            next_value = values[t]

        returns = advantages + values
        return advantages, returns


def create_aim_env(agent_role):
    return ArenaBrawlEnv(agent_role=agent_role(), opponent_role=agent_role(),
                         opponent_bot=StationaryBot, aim_practice=True)


def train_aim_a2c(agent_role=Gunner, total_steps=1_000_000, rollout_size=320,
                  num_envs=16, save_dir="ai/weights", log_path="ai/logs/a2c_aim.csv",
                  wandb_project="arena-brawl-a2c-aim", wandb_mode="online"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    envs = AsyncVectorEnv(
        [partial(create_aim_env, agent_role) for _ in range(num_envs)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    agent = A2CAgent()
    buffer = ParallelRolloutBuffer(num_envs)

    run = wandb.init(
        project=wandb_project,
        name=f"a2c-aim-{agent_role.__name__.lower()}-{num_envs}envs",
        group="a2c-aim",
        mode=wandb_mode,
        config={
            "algorithm": "a2c",
            "role": agent_role.__name__,
            "opponent": StationaryBot.__name__,
            "total_steps": total_steps,
            "rollout_size": rollout_size,
            "num_envs": num_envs,
            "gamma": agent.gamma,
            "gae_lambda": agent.gae_lambda,
            "entropy_coef": agent.entropy_coef,
            "value_coef": agent.value_coef,
        },
        reinit="finish_previous",
    )

    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["update", "steps", "avg_hit_reward", "entropy", "policy_loss", "value_loss", "episodes"])

    states, _ = envs.reset()
    episode_rewards = np.zeros(num_envs, dtype=np.float32)
    completed = []
    steps_done = 0
    update_num = 0
    start_time = last_print = time.time()
    print(f"=== AIM-A2C {agent_role.__name__} | target practice vs respawning StationaryBot "
          f"| {num_envs} envs ===")

    try:
        while steps_done < total_steps:
            rollout_steps = max(1, rollout_size // num_envs)

            for _ in range(rollout_steps):
                actions, log_probs, values = agent.select_actions(states)
                next_states, rewards, terminated, truncated, _ = envs.step(actions)
                dones = np.logical_or(terminated, truncated)

                buffer.store(states, actions, rewards, dones, log_probs, values)
                episode_rewards += rewards
                states = next_states
                steps_done += num_envs

                for index in np.flatnonzero(dones):
                    completed.append(float(episode_rewards[index]))
                    episode_rewards[index] = 0.0

                if steps_done >= total_steps:
                    break

            last_values = agent.get_values(states)
            policy_loss, value_loss, entropy = agent.update(buffer, last_values)
            buffer.clear()
            update_num += 1

            avg = sum(completed) / len(completed) if completed else 0
            writer.writerow([update_num, steps_done, round(avg, 3), round(entropy, 4),
                             round(policy_loss, 4), round(value_loss, 4), len(completed)])
            log_file.flush()

            elapsed_seconds = max(time.time() - start_time, 1e-6)
            run.log({
                "update": update_num,
                "steps": steps_done,
                "avg_hit_reward": avg,
                "entropy": entropy,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "episodes": len(completed),
                "steps_per_second": steps_done / elapsed_seconds,
            })

            if time.time() - last_print >= 30:
                elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                print(f"[{elapsed}] {agent_role.__name__} | update {update_num} | steps {steps_done} "
                      f"| avg_hit_reward {avg:.2f} | entropy {entropy:.3f} | episodes {len(completed)}")
                last_print = time.time()
            completed = []

        agent.save(os.path.join(save_dir, f"a2c_{agent_role.__name__.lower()}_aim.pth"))
    finally:
        log_file.close()
        envs.close()
        run.finish()

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"[{elapsed}] AIM-A2C {agent_role.__name__} DONE ({steps_done} steps).")


if __name__ == "__main__":
    train_aim_a2c(agent_role=Gunner, total_steps=1_000_000)

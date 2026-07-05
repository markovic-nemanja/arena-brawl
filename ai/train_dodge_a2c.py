"""DODGE TRAINING (A2C) against the same moving aimed shooter used by the other agents.

The opponent is AimShooterBot(0.7) using Gunner bullets and the environment runs in
dodge_practice mode. Reward is non-positive damage taken, so a return rising toward zero
means that the policy is learning to avoid bullets and walls.

Run: python -m ai.train_dodge_a2c
"""

import csv
import os
import time
from collections import deque

from arena_env import ArenaBrawlEnv
from systems.controller import AimShooterBot
from systems.roles import Dasher, Gunner
from ai.a2c_agent import A2CAgent
from ai.rollout_buffer import RolloutBuffer


DODGE_ROLES = [Gunner, Dasher]


def train_dodge_a2c(agent_role=Gunner, total_steps=1_000_000, rollout_size=256,
                    save_dir="ai/weights", log_path="ai/logs/a2c_dodge.csv"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    env = ArenaBrawlEnv(
        agent_role=agent_role(),
        opponent_role=Gunner(),
        opponent_bot=lambda: AimShooterBot(0.7),
        dodge_practice=True,
    )
    agent = A2CAgent(entropy_coef=0.005)
    buffer = RolloutBuffer()

    state, _ = env.reset()
    episode_reward = 0.0
    recent_returns = deque(maxlen=100)
    total_episodes = 0
    steps_done = 0
    update_num = 0
    start_time = last_print = time.time()

    print(
        f"=== DODGE-A2C {agent_role.__name__} | "
        "vs moving AimShooter(0.7) firing Gunner bullets ==="
    )

    with open(log_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow([
            "update", "steps", "avg_return_100", "step_reward", "entropy",
            "policy_loss", "value_loss", "episodes",
        ])

        try:
            while steps_done < total_steps:
                rollout_reward = 0.0
                collected_steps = 0

                for _ in range(rollout_size):
                    action, log_prob, value = agent.select_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    buffer.store(state, action, reward, done, log_prob, value)
                    episode_reward += reward
                    rollout_reward += reward
                    collected_steps += 1
                    steps_done += 1
                    state = next_state

                    if done:
                        recent_returns.append(episode_reward)
                        total_episodes += 1
                        episode_reward = 0.0
                        state, _ = env.reset()

                    if steps_done >= total_steps:
                        break

                last_value = agent.select_action(state)[2]
                policy_loss, value_loss, entropy = agent.update(buffer, last_value)
                buffer.clear()
                update_num += 1

                avg_return = (
                    sum(recent_returns) / len(recent_returns)
                    if recent_returns else 0.0
                )
                step_reward = rollout_reward / collected_steps
                writer.writerow([
                    update_num,
                    steps_done,
                    round(avg_return, 3),
                    round(step_reward, 5),
                    round(entropy, 4),
                    round(policy_loss, 4),
                    round(value_loss, 4),
                    total_episodes,
                ])
                log_file.flush()

                if time.time() - last_print >= 30:
                    elapsed = time.strftime(
                        "%H:%M:%S", time.gmtime(time.time() - start_time)
                    )
                    print(
                        f"[{elapsed}] {agent_role.__name__} | update {update_num} "
                        f"| steps {steps_done} | avg_return_100 {avg_return:.2f} "
                        f"| step_reward {step_reward:.4f} | entropy {entropy:.3f}"
                    )
                    last_print = time.time()

            agent.save(
                os.path.join(
                    save_dir, f"a2c_{agent_role.__name__.lower()}_dodge.pth"
                )
            )
        finally:
            env.close()

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"[{elapsed}] DODGE-A2C {agent_role.__name__} DONE ({steps_done} steps).")


if __name__ == "__main__":
    for role in DODGE_ROLES:
        print(f"\n######## A2C dodge: {role.__name__} ########")
        train_dodge_a2c(
            agent_role=role,
            total_steps=1_000_000,
            log_path=f"ai/logs/a2c_{role.__name__.lower()}_dodge.csv",
        )

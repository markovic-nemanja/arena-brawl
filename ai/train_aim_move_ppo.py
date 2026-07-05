"""MOVING-TARGET AIM TRAINING (PPO) - training versus MOVING target (RandomBot)
Training a PPO agent to learn to aim at respawning MOVING targets.
The agent gets rewarded only for hitting the target.
Target Respawns at a new random position every time it's hit, and has chaotic movement, so agent learns to aim at the target
Using weights from stationry aim training (ppo_<role>_aim.pth) as a seed and agent has task to learn to predict movement and aim based on it

Expceting the best performance from PPO
"""
import os
import csv
import time

from arena_env import ArenaBrawlEnv
from systems.roles import *
from systems.controller import RandomBot
from ai.rollout_buffer import RolloutBuffer
from ai.ppo_agent import PPOAgent


def train_aim_move(agent_role=Gunner, total_steps=1_000_000, rollout_size=2048,
                   save_dir="ai/weights", log_path="ai/logs/ppo_aim_move.csv"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # MOVING target: a RandomBot drifts around the arena, and respawns at a new spot on each hit
    env = ArenaBrawlEnv(agent_role=agent_role(), opponent_role=agent_role(),
                        opponent_bot=RandomBot, aim_practice=True)

    agent = PPOAgent()
    seed = os.path.join(save_dir, f"ppo_{agent_role.__name__.lower()}_aim.pth")   # TRANSFER from the stationary seed
    agent.load(seed)
    print(f"[transfer] loaded stationary seed: {seed}")

    buffer = RolloutBuffer()
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["update", "steps", "avg_hit_reward", "entropy", "policy_loss", "value_loss", "episodes"])

    state, _ = env.reset()
    episode_reward = 0
    completed = []
    steps_done = 0
    update_num = 0
    start_time = last_print = time.time()
    print(f"=== AIM-MOVE {agent_role.__name__} | vs MOVING RandomBot (transfer from stationary seed) ===")

    while steps_done < total_steps:
        for _ in range(rollout_size):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.store(state, action, reward, done, log_prob, value)
            episode_reward += reward
            state = next_state
            steps_done += 1
            if done:
                completed.append(episode_reward)
                episode_reward = 0
                state, _ = env.reset()
            if steps_done >= total_steps:
                break

        last_value = agent.select_action(state)[2]
        policy_loss, value_loss, entropy = agent.update(buffer, last_value)
        buffer.clear()
        update_num += 1

        avg = sum(completed) / len(completed) if completed else 0
        writer.writerow([update_num, steps_done, round(avg, 3), round(entropy, 4),
                         round(policy_loss, 4), round(value_loss, 4), len(completed)])
        log_file.flush()

        if time.time() - last_print >= 30:
            elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            print(f"[{elapsed}] {agent_role.__name__} | update {update_num} | steps {steps_done} "
                  f"| avg_hit_reward {avg:.2f} | entropy {entropy:.3f} | episodes {len(completed)}")
            last_print = time.time()
        completed = []

    agent.save(os.path.join(save_dir, f"ppo_{agent_role.__name__.lower()}_aim_move.pth"))
    log_file.close()
    env.close()
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"[{elapsed}] AIM-MOVE {agent_role.__name__} DONE ({steps_done} steps).")


if __name__ == "__main__":
    for role in [Gunner, Bomber, Dasher, ToxicTrail, Blackhole]:
        print(f"\n######## PPO moving-aim: {role.__name__} ########")
        train_aim_move(agent_role=role, total_steps=1_000_000,
                       log_path=f"ai/logs/ppo_{role.__name__.lower()}_aim_move.csv")

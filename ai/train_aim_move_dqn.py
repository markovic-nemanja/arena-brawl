"""MOVING-TARGET AIM TRAINING (DQN) - training versus MOVING target (RandomBot)
Training a DQN agent to learn to aim at respawning MOVING targets.
The agent gets rewarded only for hitting the target.
Target Respawns at a new random position every time it's hit, and has chaotic movement, so agent learns to aim at the target
Using weights from stationry aim training (dqn_<role>_aim.pth) as a seed and agent has task to learn to predict movement and aim based on it
"""
import os
import csv
import time

from arena_env import ArenaBrawlEnv
from systems.roles import *
from systems.controller import RandomBot
from ai.replay_buffer import ReplayBuffer
from ai.dqn_agent import DQNAgent


def train_aim_move_dqn(agent_role=Gunner, total_steps=1_000_000, buffer_capacity=100000, batch_size=64,
                       save_dir="ai/weights", log_path="ai/logs/dqn_aim_move.csv"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # MOVING target: a RandomBot drifts around the arena, and respawns at a new spot on each hit
    env = ArenaBrawlEnv(agent_role=agent_role(), opponent_role=agent_role(),
                        opponent_bot=RandomBot, aim_practice=True)

    agent = DQNAgent(batch_size=batch_size, epsilon_start=0.3)   # low start: exploit the seed, don't erase it
    seed = os.path.join(save_dir, f"dqn_{agent_role.__name__.lower()}_aim.pth") # TRANSFER from the stationary seed
    agent.load(seed)
    print(f"[transfer] loaded stationary seed: {seed}")

    buffer = ReplayBuffer(capacity=buffer_capacity)
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["episode", "steps", "hit_reward", "epsilon", "loss"])

    steps_done = 0
    episode = 0
    start_time = last_print = time.time()
    print(f"=== AIM-MOVE-DQN {agent_role.__name__} | vs MOVING RandomBot (transfer from stationary seed) ===")

    while steps_done < total_steps:
        state, _ = env.reset()
        ep_reward = 0.0
        total_loss = 0.0
        loss_count = 0
        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.store(state, action, reward, next_state, float(done))
            loss = agent.train_step(buffer)
            if loss is not None:
                total_loss += loss
                loss_count += 1
            ep_reward += reward
            state = next_state
            steps_done += 1
            if done or steps_done >= total_steps:
                break

        agent.decay_epsilon()
        episode += 1
        avg_loss = total_loss / loss_count if loss_count > 0 else 0
        writer.writerow([episode, steps_done, round(ep_reward, 3), round(agent.epsilon, 4), round(avg_loss, 5)])
        log_file.flush()

        if time.time() - last_print >= 30:
            elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            print(f"[{elapsed}] {agent_role.__name__} | ep {episode} | steps {steps_done} "
                  f"| hit_reward {ep_reward:.0f} | eps {agent.epsilon:.3f} | loss {avg_loss:.4f}")
            last_print = time.time()

    agent.save(os.path.join(save_dir, f"dqn_{agent_role.__name__.lower()}_aim_move.pth"))
    log_file.close()
    env.close()
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"[{elapsed}] AIM-MOVE-DQN {agent_role.__name__} DONE ({steps_done} steps).")


if __name__ == "__main__":
    for role in [Gunner, Bomber, Dasher, ToxicTrail, Blackhole]:
        print(f"\n######## DQN moving-aim: {role.__name__} ########")
        train_aim_move_dqn(agent_role=role, total_steps=1_000_000,
                           log_path=f"ai/logs/dqn_{role.__name__.lower()}_aim_move.csv")

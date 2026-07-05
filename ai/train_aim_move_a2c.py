"""MOVING-TARGET AIM TRAINING (A2C, single-env) — stage 2 of the aim ladder, no parallelization.

Same as train_aim_a2c.py but the target MOVES (RandomBot), and it starts from the STATIONARY A2C aim
seed (a2c_<role>_aim.pth) and fine-tunes — transfer learning — so it refines "aim at a fixed spot" into
"aim at a moving target". A2C is on-policy and explores through policy entropy (no epsilon), so transfer
is simply: load the seed weights and keep training against the harder, moving opponent.

"""
import os
import csv
import time
from collections import deque

from arena_env import ArenaBrawlEnv
from systems.roles import *
from systems.controller import RandomBot
from ai.rollout_buffer import RolloutBuffer
from ai.a2c_agent import A2CAgent


def train_aim_move_a2c(agent_role=Gunner, total_steps=1_000_000, rollout_size=256,
                       save_dir="ai/weights", log_path="ai/logs/a2c_aim_move.csv"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # MOVING target: a RandomBot drifts around the arena, and respawns at a new spot on each hit
    env = ArenaBrawlEnv(agent_role=agent_role(), opponent_role=agent_role(),
                        opponent_bot=RandomBot, aim_practice=True)

    agent = A2CAgent()
    seed = os.path.join(save_dir, f"a2c_{agent_role.__name__.lower()}_aim.pth")   # TRANSFER from the stationary seed
    agent.load(seed)
    print(f"[transfer] loaded stationary seed: {seed}")

    buffer = RolloutBuffer()

    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["update", "steps", "avg_return_100", "step_reward",
                     "entropy", "policy_loss", "value_loss", "episodes"])

    state, _ = env.reset()
    episode_reward = 0.0
    recent_returns = deque(maxlen=100)   # rolling window of completed-episode returns (never 0-inflated)
    total_episodes = 0
    steps_done = 0
    update_num = 0
    start_time = last_print = time.time()
    print(f"=== AIM-MOVE-A2C {agent_role.__name__} | vs MOVING RandomBot (transfer from stationary seed) ===")

    while steps_done < total_steps:
        rollout_reward = 0.0
        rollout_steps = 0
        for _ in range(rollout_size):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.store(state, action, reward, done, log_prob, value)
            episode_reward += reward
            rollout_reward += reward
            state = next_state
            steps_done += 1
            rollout_steps += 1
            if done:
                recent_returns.append(episode_reward)
                total_episodes += 1
                episode_reward = 0.0
                state, _ = env.reset()
            if steps_done >= total_steps:
                break

        last_value = agent.select_action(state)[2]        # bootstrap value for the tail
        policy_loss, value_loss, entropy = agent.update(buffer, last_value)
        buffer.clear()
        update_num += 1

        avg_return = sum(recent_returns) / len(recent_returns) if recent_returns else 0.0
        step_reward = rollout_reward / rollout_steps       # mean reward per step (never zero-inflated)
        writer.writerow([update_num, steps_done, round(avg_return, 3), round(step_reward, 5),
                         round(entropy, 4), round(policy_loss, 4), round(value_loss, 4), total_episodes])
        log_file.flush()

        if time.time() - last_print >= 30:
            elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            print(f"[{elapsed}] {agent_role.__name__} | update {update_num} | steps {steps_done} "
                  f"| avg_return_100 {avg_return:.2f} | step_reward {step_reward:.4f} | entropy {entropy:.3f}")
            last_print = time.time()

    agent.save(os.path.join(save_dir, f"a2c_{agent_role.__name__.lower()}_aim_move.pth"))
    log_file.close()
    env.close()
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"[{elapsed}] AIM-MOVE-A2C {agent_role.__name__} DONE ({steps_done} steps).")


if __name__ == "__main__":
    for role in [Gunner, Bomber, Dasher, ToxicTrail, Blackhole]:
        print(f"\n######## A2C moving-aim: {role.__name__} ########")
        train_aim_move_a2c(agent_role=role, total_steps=1_000_000,
                           log_path=f"ai/logs/a2c_{role.__name__.lower()}_aim_move.csv")
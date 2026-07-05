"""AIM TRAINING (A2C, single-env) — mirrors train_aim_ppo.py.

Same isolated aiming task: a StationaryBot respawns at a new random spot on each hit, reward is
damage-dealt only, so the agent must read the target's position and aim. This is the single-environment
A2C counterpart to train_aim_ppo.py — identical env / rollout / step budget, only the algorithm's update
differs (A2C = one policy-gradient step per rollout, no PPO clipping or multi-epoch reuse). That makes it
a clean apples-to-apples comparison with PPO.

"""
import os
import csv
import time
from collections import deque

from arena_env import ArenaBrawlEnv
from systems.roles import *
from systems.controller import StationaryBot
from ai.rollout_buffer import RolloutBuffer
from ai.a2c_agent import A2CAgent


def train_aim_a2c(agent_role=Gunner, total_steps=1_000_000, rollout_size=256,
                  save_dir="ai/weights", log_path="ai/logs/a2c_aim.csv"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    env = ArenaBrawlEnv(agent_role=agent_role(), opponent_role=agent_role(),
                        opponent_bot=StationaryBot, aim_practice=True)
    agent = A2CAgent()
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
    print(f"=== AIM-A2C {agent_role.__name__} | target practice vs respawning StationaryBot (single env) ===")

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

    agent.save(os.path.join(save_dir, f"a2c_{agent_role.__name__.lower()}_aim.pth"))
    log_file.close()
    env.close()
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"[{elapsed}] AIM-A2C {agent_role.__name__} DONE ({steps_done} steps).")


if __name__ == "__main__":
    for role in [Gunner, Bomber, Dasher, ToxicTrail, Blackhole]:
        print(f"\n######## A2C aim: {role.__name__} ########")
        train_aim_a2c(agent_role=role, total_steps=1_000_000,
                      log_path=f"ai/logs/a2c_{role.__name__.lower()}_aim.csv")
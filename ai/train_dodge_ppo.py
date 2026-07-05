"""DODGE TRAINING (PPO) — Training versus moving hazards-.

Training is done only on Gunner and Dasher.
The opponent is a moving AimShooterBot that fires Gunner bullets since they are hardest do doge.
Bomber, ToxicTrail, and Blackhole should have the same dodge policy as Gunner.
Dasher can use ability to dodge (huge success if he actually does this).

Super bonus if he learns to dodge walls, since they are dealing damage too.

"""
import os
import csv
import time

from arena_env import ArenaBrawlEnv
from systems.roles import *
from systems.controller import AimShooterBot
from ai.rollout_buffer import RolloutBuffer
from ai.ppo_agent import PPOAgent


def train_dodge(agent_role=Gunner, total_steps=1_000_000, rollout_size=2048,
                save_dir="ai/weights", log_path="ai/logs/ppo_dodge.csv"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # opponent = a MOVING shooter that aims ~70% of its bursts and fires Gunner bullets; 
    # moving (vs stationary turret) means the agent can't escape by running out of range — it has to dodge.
    env = ArenaBrawlEnv(agent_role=agent_role(), opponent_role=Gunner(),
                        opponent_bot=lambda: AimShooterBot(0.7), dodge_practice=True)

    agent = PPOAgent()
    buffer = RolloutBuffer()

    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["update", "steps", "avg_dodge_reward", "entropy", "policy_loss", "value_loss", "episodes"])

    state, _ = env.reset()
    episode_reward = 0
    completed = []
    steps_done = 0
    update_num = 0
    start_time = last_print = time.time()
    print(f"=== DODGE {agent_role.__name__} | vs moving AimShooter(0.7) firing Gunner bullets ===")

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
                  f"| avg_dodge_reward {avg:.2f} | entropy {entropy:.3f} | episodes {len(completed)}")
            last_print = time.time()
        completed = []

    agent.save(os.path.join(save_dir, f"ppo_{agent_role.__name__.lower()}_dodge.pth"))
    log_file.close()
    env.close()
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"[{elapsed}] DODGE {agent_role.__name__} DONE ({steps_done} steps).")


if __name__ == "__main__":
    for role in [Gunner, Bomber, Dasher, ToxicTrail, Blackhole]:
        print(f"\n######## PPO dodge: {role.__name__} ########")
        train_dodge(agent_role=role, total_steps=1_000_000,
                    log_path=f"ai/logs/ppo_{role.__name__.lower()}_dodge.csv")

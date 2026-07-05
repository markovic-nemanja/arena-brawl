"""AIM TRAINING (DQN) - training versus stationary target (StationaryBot)
Training a DQN agent to learn to aim at respawning STATIONARY targets.
The agent gets rewarded only for hitting the target,
Target Respawns at a new random position every time it's hit, so agent learns to aim at the target
Part of partial tasks to see if agent acuatlly can learn
"""
import os
import csv
import time

from arena_env import ArenaBrawlEnv
from systems.roles import *
from systems.controller import StationaryBot
from ai.replay_buffer import ReplayBuffer
from ai.dqn_agent import DQNAgent

EPS_END = 0.05 # floor: keep 5% exploration even once fully annealed


def train_aim_dqn(agent_role=Gunner, total_steps=1_000_000, opponent_bot=StationaryBot,
                  base_weights=None, eps_start=1.0, eps_anneal_frac=0.6,
                  buffer_capacity=100000, batch_size=64,
                  save_dir="ai/weights", log_path="ai/logs/dqn_aim.csv", save_suffix="aim"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    env = ArenaBrawlEnv(agent_role=agent_role(), opponent_role=agent_role(),
                        opponent_bot=opponent_bot, aim_practice=True)
    agent = DQNAgent(batch_size=batch_size)
    if base_weights: # transfer: start from an earlier seed instead of scratch
        agent.load(base_weights)
        print(f"[transfer] loaded {base_weights}")
    buffer = ReplayBuffer(capacity=buffer_capacity)

    # STEP-BASED LINEAR epsilon anneal: eps_start -> EPS_END over the first eps_anneal_frac of steps,
    # then hold at EPS_END. Tied to STEPS (not episodes) so the schedule is predictable regardless of
    # episode length — the old per-episode 0.999 decay only reached ~0.48 in 1M steps (never exploited).
    # Moving/transfer runs pass a low eps_start (e.g. 0.3) so exploration doesn't wipe the loaded seed.
    anneal_steps = max(1, int(total_steps * eps_anneal_frac))
    agent.epsilon = eps_start

    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["episode", "steps", "hit_reward", "epsilon", "loss"])

    steps_done = 0
    episode = 0
    start_time = last_print = time.time()
    print(f"=== AIM-DQN {agent_role.__name__} vs {opponent_bot.__name__} (target practice) ===")

    while steps_done < total_steps:
        state, _ = env.reset()
        ep_reward = 0.0
        total_loss = 0.0
        loss_count = 0
        while True:
            agent.epsilon = max(EPS_END, eps_start - (eps_start - EPS_END) * steps_done / anneal_steps)
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

        # epsilon is scheduled per-step above (no per-episode decay) - not exploring firing problem solution
        episode += 1
        avg_loss = total_loss / loss_count if loss_count > 0 else 0
        writer.writerow([episode, steps_done, round(ep_reward, 3), round(agent.epsilon, 4), round(avg_loss, 5)])
        log_file.flush()

        if time.time() - last_print >= 30:
            elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            print(f"[{elapsed}] {agent_role.__name__} | ep {episode} | steps {steps_done} "
                  f"| hit_reward {ep_reward:.0f} | eps {agent.epsilon:.3f} | loss {avg_loss:.4f}")
            last_print = time.time()

    agent.save(os.path.join(save_dir, f"dqn_{agent_role.__name__.lower()}_{save_suffix}.pth"))
    log_file.close()
    env.close()
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"[{elapsed}] AIM-DQN {agent_role.__name__} DONE ({steps_done} steps).")


if __name__ == "__main__":
    for role in [Gunner, Bomber, Dasher, ToxicTrail, Blackhole]:
        print(f"\n######## DQN aim: {role.__name__} ########")
        train_aim_dqn(agent_role=role, total_steps=1_000_000,
                      log_path=f"ai/logs/dqn_{role.__name__.lower()}_aim.csv")

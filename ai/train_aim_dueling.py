"""AIM TRAINING (Dueling DQN) — value-based counterpart with the dueling network.

Same isolated aiming task as train_aim.py (PPO) / train_aim_dqn.py (DQN): a respawning target,
hit-only reward, aim_practice env. For the algorithm comparison and to get Dueling aim seeds.

`opponent_bot` + `base_weights` are exposed so the same script does the ladder (stationary -> moving,
with transfer). Run:  python -m ai.train_aim_dueling
"""
import os
import csv
import time

from arena_env import ArenaBrawlEnv
from systems.roles import *
from systems.controller import StationaryBot
from ai.replay_buffer import ReplayBuffer
from ai.dueling_dqn_agent import DuelingDQNAgent

EPS_END = 0.05   # floor: keep 5% exploration even once fully annealed


def train_aim_dueling(agent_role=Gunner, total_steps=1_000_000, opponent_bot=StationaryBot,
                      base_weights=None, eps_start=1.0, eps_anneal_frac=0.6,
                      buffer_capacity=100000, batch_size=64,
                      save_dir="ai/weights", log_path="ai/logs/dueling_aim.csv", save_suffix="aim"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    env = ArenaBrawlEnv(agent_role=agent_role(), opponent_role=agent_role(),
                        opponent_bot=opponent_bot, aim_practice=True)
    agent = DuelingDQNAgent(state_size=22, batch_size=batch_size)   # 22-vec obs (friend's default may still be 20)
    if base_weights:                       # transfer: start from an earlier seed instead of scratch
        agent.load(base_weights)
        print(f"[transfer] loaded {base_weights}")
    buffer = ReplayBuffer(capacity=buffer_capacity)

    # STEP-BASED LINEAR epsilon anneal (identical schedule to train_aim_dqn so DQN vs Dueling differs by
    # ARCHITECTURE, not exploration): eps_start -> EPS_END over the first eps_anneal_frac of steps, then
    # hold. Tied to steps, not episodes — the old per-episode 0.999 decay only reached ~0.48 in 1M steps.
    anneal_steps = max(1, int(total_steps * eps_anneal_frac))
    agent.epsilon = eps_start

    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["episode", "steps", "hit_reward", "epsilon", "loss"])

    steps_done = 0
    episode = 0
    start_time = last_print = time.time()
    print(f"=== AIM-DUELING {agent_role.__name__} vs {opponent_bot.__name__} (target practice) ===")

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

        # epsilon is scheduled per-step above (no per-episode decay)
        episode += 1
        avg_loss = total_loss / loss_count if loss_count > 0 else 0
        writer.writerow([episode, steps_done, round(ep_reward, 3), round(agent.epsilon, 4), round(avg_loss, 5)])
        log_file.flush()

        if time.time() - last_print >= 30:
            elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            print(f"[{elapsed}] {agent_role.__name__} | ep {episode} | steps {steps_done} "
                  f"| hit_reward {ep_reward:.0f} | eps {agent.epsilon:.3f} | loss {avg_loss:.4f}")
            last_print = time.time()

    agent.save(os.path.join(save_dir, f"dueling_{agent_role.__name__.lower()}_{save_suffix}.pth"))
    log_file.close()
    env.close()
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"[{elapsed}] AIM-DUELING {agent_role.__name__} DONE ({steps_done} steps).")


if __name__ == "__main__":
    for role in [Gunner, Bomber, Dasher, ToxicTrail, Blackhole]:
        print(f"\n######## Dueling aim: {role.__name__} ########")
        train_aim_dueling(agent_role=role, total_steps=1_000_000,
                          log_path=f"ai/logs/dueling_{role.__name__.lower()}_aim.csv")

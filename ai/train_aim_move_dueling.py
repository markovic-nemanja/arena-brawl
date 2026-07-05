"""MOVING-TARGET AIM TRAINING (Dueling DQN) — stage 2 of the aim ladder, dueling variant.

Same as train_aim_dueling.py but the target MOVES (RandomBot), and it starts from the STATIONARY
Dueling aim seed (dueling_<role>_aim.pth) and fine-tunes (transfer). NOTE: epsilon starts LOW (0.3), not
1.0 — a fresh 1.0 would flood the buffer with random actions and erase the seed; we want it to mostly
EXPLOIT the seed and explore only a little for the new moving dynamics.

Mirrors train_aim_move_dqn.py (same schedule) so the DQN vs Dueling moving comparison is fair.

Run:  python -m ai.train_aim_move_dueling   (train ai.train_aim_dueling first — it produces the seed this loads)
"""
import os
import csv
import time

from arena_env import ArenaBrawlEnv
from systems.roles import *
from systems.controller import RandomBot
from ai.replay_buffer import ReplayBuffer
from ai.dueling_dqn_agent import DuelingDQNAgent

EPS_END = 0.05   # floor: keep 5% exploration even once fully annealed


def train_aim_move_dueling(agent_role=Gunner, total_steps=1_000_000, eps_start=0.3, eps_anneal_frac=0.6,
                           buffer_capacity=100000, batch_size=64,
                           save_dir="ai/weights", log_path="ai/logs/dueling_aim_move.csv"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # MOVING target: a RandomBot drifts around the arena, and respawns at a new spot on each hit
    env = ArenaBrawlEnv(agent_role=agent_role(), opponent_role=agent_role(),
                        opponent_bot=RandomBot, aim_practice=True)

    # low epsilon start: exploit the seed, don't erase it (22-vec obs, so force state_size=22)
    agent = DuelingDQNAgent(state_size=22, batch_size=batch_size, epsilon_start=eps_start)
    seed = os.path.join(save_dir, f"dueling_{agent_role.__name__.lower()}_aim.pth")   # TRANSFER from the stationary seed
    agent.load(seed)
    print(f"[transfer] loaded stationary seed: {seed}")

    buffer = ReplayBuffer(capacity=buffer_capacity)

    # STEP-BASED LINEAR epsilon anneal (matches train_aim_move_dqn.py): eps_start -> EPS_END over the
    # first eps_anneal_frac of steps, then hold — the old per-episode 0.999 decay only crawled 0.3 -> ~0.14.
    anneal_steps = max(1, int(total_steps * eps_anneal_frac))
    agent.epsilon = eps_start

    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["episode", "steps", "hit_reward", "epsilon", "loss"])

    steps_done = 0
    episode = 0
    start_time = last_print = time.time()
    print(f"=== AIM-MOVE-DUELING {agent_role.__name__} | vs MOVING RandomBot (transfer from stationary seed) ===")

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

    agent.save(os.path.join(save_dir, f"dueling_{agent_role.__name__.lower()}_aim_move.pth"))
    log_file.close()
    env.close()
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"[{elapsed}] AIM-MOVE-DUELING {agent_role.__name__} DONE ({steps_done} steps).")


if __name__ == "__main__":
    for role in [Gunner, Bomber, Dasher, ToxicTrail, Blackhole]:
        print(f"\n######## Dueling moving-aim: {role.__name__} ########")
        train_aim_move_dueling(agent_role=role, total_steps=1_000_000,
                               log_path=f"ai/logs/dueling_{role.__name__.lower()}_aim_move.csv")
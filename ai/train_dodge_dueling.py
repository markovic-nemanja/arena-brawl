"""DODGE TRAINING (Dueling DQN) — value-based counterpart with the dueling network.

Same isolated dodging task as train_dodge_ppo.py / train_dodge_dqn.py: a moving aimed shooter
(AimShooterBot at 0.7) fires Gunner bullets at the agent, both fighters are immortal, and the reward is
-damage TAKEN only (projectile hits AND wall contact). For the algorithm comparison and dodge seeds.

Epsilon uses the SAME step-based linear anneal as train_dodge_dqn.py so DQN vs Dueling differ by
architecture, not exploration.

Reward note: it is <= 0 (0 = never hit). A rising dodge_reward toward 0 means it is learning to dodge.

Run:  python -m ai.train_dodge_dueling
"""
import os
import csv
import time

from arena_env import ArenaBrawlEnv
from systems.roles import *
from systems.controller import AimShooterBot
from ai.replay_buffer import ReplayBuffer
from ai.dueling_dqn_agent import DuelingDQNAgent

EPS_END = 0.05   # floor: keep 5% exploration even once fully annealed


def train_dodge_dueling(agent_role=Gunner, total_steps=1_000_000, base_weights=None,
                        eps_start=1.0, eps_anneal_frac=0.6, buffer_capacity=100000, batch_size=64,
                        save_dir="ai/weights", log_path="ai/logs/dueling_dodge.csv", save_suffix="dodge"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # opponent = a MOVING shooter that aims ~70% of its bursts and fires Gunner bullets; moving (vs a
    # stationary turret) means the agent can't escape by running out of range — it has to dodge.
    env = ArenaBrawlEnv(agent_role=agent_role(), opponent_role=Gunner(),
                        opponent_bot=lambda: AimShooterBot(0.7), dodge_practice=True)
    agent = DuelingDQNAgent(state_size=22, batch_size=batch_size) 
    if base_weights:                    
        agent.load(base_weights)
        print(f"[transfer] loaded {base_weights}")
    buffer = ReplayBuffer(capacity=buffer_capacity)

    # STEP-BASED LINEAR epsilon anneal (identical schedule to train_dodge_dqn.py so the comparison is fair).
    anneal_steps = max(1, int(total_steps * eps_anneal_frac))
    agent.epsilon = eps_start

    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["episode", "steps", "dodge_reward", "epsilon", "loss"])

    steps_done = 0
    episode = 0
    start_time = last_print = time.time()
    print(f"=== DODGE-DUELING {agent_role.__name__} vs moving AimShooter(0.7) firing Gunner bullets ===")

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
                  f"| dodge_reward {ep_reward:.0f} | eps {agent.epsilon:.3f} | loss {avg_loss:.4f}")
            last_print = time.time()

    agent.save(os.path.join(save_dir, f"dueling_{agent_role.__name__.lower()}_{save_suffix}.pth"))
    log_file.close()
    env.close()
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"[{elapsed}] DODGE-DUELING {agent_role.__name__} DONE ({steps_done} steps).")


if __name__ == "__main__":
    for role in [Gunner, Bomber, Dasher, ToxicTrail, Blackhole]:
        print(f"\n######## Dueling dodge: {role.__name__} ########")
        train_dodge_dueling(agent_role=role, total_steps=1_000_000,
                            log_path=f"ai/logs/dueling_{role.__name__.lower()}_dodge.csv")

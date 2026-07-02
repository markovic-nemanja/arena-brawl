import os
import csv
import time
import random
from functools import partial
from collections import deque

import wandb
from arena_env import ArenaBrawlEnv
from systems.roles import *
from systems.controller import (StationaryBot, RandomBot, RandomShooterBot,
                                 GentleAggressor, EasyBot, MediumBot)
from ai.replay_buffer import ReplayBuffer
from ai.dqn_agent import DQNAgent

ALL_ROLES = [Gunner, Bomber, Dasher, ToxicTrail, Blackhole]

# Ramped curriculum. GentleAggressor dials smoothly up to EasyBot (fire_prob 1.0);
# HardBot is held out as the eval benchmark, not a training rung.
PHASE_BOTS = {
    1: StationaryBot,
    2: RandomBot,
    3: RandomShooterBot,
    4: partial(GentleAggressor, fire_prob=0.4),
    5: partial(GentleAggressor, fire_prob=0.7),
    6: EasyBot,
    7: MediumBot,
}
PHASE_BOUNDS = (200_000, 400_000, 650_000, 950_000, 1_250_000, 1_600_000)

def get_phase(steps):
    phase = 1
    for b in PHASE_BOUNDS:
        if steps >= b:
            phase += 1
        else:
            break
    return phase

def _bot_name(phase):
    b = PHASE_BOTS[phase]
    return b.func.__name__ if isinstance(b, partial) else b.__name__

def train_dqn(agent_role=Gunner, total_steps=2_000_000, buffer_capacity=100000,
              batch_size=64, save_dir="ai/weights", log_path="ai/logs/dqn_training.csv",
              checkpoint_every=500_000):

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    env = ArenaBrawlEnv(agent_role=agent_role())
    agent = DQNAgent(batch_size=batch_size)
    buffer = ReplayBuffer(capacity=buffer_capacity)

    wandb.init(project="arena-brawl-dqn", name=agent_role.__name__,
               config={"total_steps": total_steps, "batch_size": batch_size,
                       "buffer_capacity": buffer_capacity, "gamma": agent.gamma,
                       "epsilon_decay": agent.epsilon_decay, "phase_bounds": PHASE_BOUNDS},
               reinit=True)

    log_file = open(log_path, mode='w', newline='')
    writer = csv.writer(log_file)
    writer.writerow(["episode", "steps", "phase", "reward", "win", "epsilon", "loss"])

    def set_opponent(phase):
        env.opponent_role = random.choice(ALL_ROLES)()  # random body -> all hazard types appear
        env.opponent_bot  = PHASE_BOTS[phase]

    last_phase = get_phase(0)
    set_opponent(last_phase)

    steps_done = 0
    episode = 0
    last_checkpoint = 0
    recent_wins = deque(maxlen=100)
    start_time = last_print = time.time()
    print(f"=== DQN {agent_role.__name__} | Phase {last_phase} ({_bot_name(last_phase)}) ===")

    while steps_done < total_steps:
        state, _ = env.reset()
        total_reward = 0.0
        total_loss = 0.0
        loss_count = 0
        win = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.store(state, action, reward, next_state, float(done))

            loss = agent.train_step(buffer)
            if loss is not None:
                total_loss += loss
                loss_count += 1

            total_reward += reward
            state = next_state
            steps_done += 1

            if done or steps_done >= total_steps:
                win = 1 if env.opponent.hp <= 0 else 0
                break

        agent.decay_epsilon()
        episode += 1

        phase = get_phase(steps_done)
        if phase != last_phase:
            agent.epsilon = max(agent.epsilon, 0.5)   # re-explore against the harder bot
            elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            print(f"[{elapsed}] === {agent_role.__name__} | Phase {phase} ({_bot_name(phase)}) @ step {steps_done} ===")
            last_phase = phase
        set_opponent(phase)

        avg_loss = total_loss / loss_count if loss_count > 0 else 0
        writer.writerow([episode, steps_done, phase, round(total_reward, 3), win,
                         round(agent.epsilon, 4), round(avg_loss, 5)])
        log_file.flush()

        recent_wins.append(win)
        wandb.log({"episode": episode, "steps": steps_done, "phase": phase,
                   "reward": total_reward, "win": win,
                   "win_rate": sum(recent_wins) / len(recent_wins),
                   "epsilon": agent.epsilon, "loss": avg_loss})

        if steps_done // checkpoint_every > last_checkpoint:
            last_checkpoint = steps_done // checkpoint_every
            agent.save(os.path.join(save_dir, f"dqn_{agent_role.__name__.lower()}_{steps_done}.pth"))

        if time.time() - last_print >= 30:
            elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            print(f"[{elapsed}] {agent_role.__name__} | ep {episode} | Phase{phase} ({_bot_name(phase)}) "
                  f"| steps {steps_done} | reward {total_reward:.1f} | win {win} "
                  f"| eps {agent.epsilon:.3f} | loss {avg_loss:.5f}")
            last_print = time.time()

    agent.save(os.path.join(save_dir, f"dqn_{agent_role.__name__.lower()}_final.pth"))
    log_file.close()
    env.close()
    wandb.finish()
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"[{elapsed}] DQN {agent_role.__name__} DONE ({steps_done} steps).")

if __name__ == "__main__":
    for role in ALL_ROLES:
        print(f"\n######## DQN training: {role.__name__} ########")
        try:
            train_dqn(agent_role=role, total_steps=2_000_000,
                      log_path=f"ai/logs/dqn_{role.__name__.lower()}.csv")
        except Exception as e:
            print(f"!!! {role.__name__} FAILED: {e} !!!")

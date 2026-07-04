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
                                 AimShooterBot, GentleAggressor, GentleMedium)
from ai.replay_buffer import ReplayBuffer
from ai.dqn_agent import DQNAgent

ALL_ROLES = [Gunner, Bomber, Dasher, ToxicTrail, Blackhole]

PHASE_BOTS = {
    1: StationaryBot,
    2: RandomBot,
    3: RandomShooterBot,
    4: partial(AimShooterBot, aim_prob=0.5),
    5: partial(GentleAggressor, fire_prob=0.5),
    6: partial(GentleMedium, fire_prob=0.5),
}
GRADUATION_PHASE = max(PHASE_BOTS) + 1             # final phase: a RANDOM MIX of all combat bots, so the
GRAD_BOTS = [PHASE_BOTS[k] for k in (3, 4, 5, 6)]  # big final chunk stays GENERAL instead of overfitting one
MAX_PHASE = GRADUATION_PHASE

PHASE_MIN = {1:  50_000, 2:  65_000, 3:  85_000, 4: 100_000, 5: 100_000, 6: 100_000} # Minimum amount of steps in phase before promotion is allowed
PHASE_MAX = {1: 150_000, 2: 200_000, 3: 250_000, 4: 300_000, 5: 300_000, 6: 300_000} # Maximum amount of steps in phase before forced promotion
PHASE_PROMOTE = {1: 0.85, 2: 0.70, 3: 0.65, 4: 0.55, 5: 0.55, 6: 0.55} # Win-rate threshold for promotion (win-rate over the last WIN_WINDOW episodes)
WIN_WINDOW = 50 # how many episodes the win-rate is measured over
GRAD_STEPS = 700_000 # how many steps last pool run has

def _bot_name(phase):
    """Used for console logging"""
    if phase == GRADUATION_PHASE:
        return "Graduation(mix)"
    b = PHASE_BOTS[phase]
    return b.func.__name__ if isinstance(b, partial) else b.__name__

def train_dqn(agent_role=Gunner, total_steps=2_300_000, buffer_capacity=100000,
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
                       "epsilon_decay": agent.epsilon_decay, "phase_promote": PHASE_PROMOTE},
               reinit=True)

    log_file = open(log_path, mode='w', newline='')
    writer = csv.writer(log_file)
    writer.writerow(["episode", "steps", "phase", "reward", "win", "epsilon", "loss"])

    def set_opponent(phase):
        env.opponent_role = random.choice(ALL_ROLES)()  # random body -> all hazard types appear
        env.opponent_bot  = random.choice(GRAD_BOTS) if phase == GRADUATION_PHASE else PHASE_BOTS[phase]

    phase = 1
    phase_start_steps = 0
    finished = False
    set_opponent(phase)

    steps_done = 0
    episode = 0
    last_checkpoint = 0
    recent_wins = deque(maxlen=WIN_WINDOW)
    start_time = last_print = time.time()
    print(f"=== DQN {agent_role.__name__} | Phase {phase} ({_bot_name(phase)}) ===")

    while steps_done < total_steps and not finished:
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
                win = 1 if (terminated and env.agent.hp > 0) else 0   # alive at termination == we won
                break

        agent.decay_epsilon()
        episode += 1
        recent_wins.append(win)

        # Promote on win-rate mastery once past the floor; force-promote at the ceiling.
        if phase < MAX_PHASE:
            in_phase = steps_done - phase_start_steps
            wr = sum(recent_wins) / len(recent_wins) if recent_wins else 0
            mastered = in_phase >= PHASE_MIN[phase] and len(recent_wins) >= WIN_WINDOW and wr >= PHASE_PROMOTE[phase]
            if mastered or in_phase >= PHASE_MAX[phase]:
                phase += 1
                phase_start_steps = steps_done
                recent_wins.clear()
                agent.epsilon = max(agent.epsilon, 0.5) # re-explore against the new opponent
                why = "mastered" if mastered else "cap"
                elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                print(f"[{elapsed}] === {agent_role.__name__} | Phase {phase} ({_bot_name(phase)}) @ step {steps_done} ({why}) ===")
        elif steps_done - phase_start_steps >= GRAD_STEPS:
            finished = True # graduation done — end the run
        set_opponent(phase)

        avg_loss = total_loss / loss_count if loss_count > 0 else 0
        writer.writerow([episode, steps_done, phase, round(total_reward, 3), win,
                         round(agent.epsilon, 4), round(avg_loss, 5)])
        log_file.flush()

        wandb.log({"episode": episode, "steps": steps_done, "phase": phase,
                   "reward": total_reward, "win": win,
                   "win_rate": sum(recent_wins) / len(recent_wins) if recent_wins else 0,
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
            train_dqn(agent_role=role, total_steps=2_300_000,
                      log_path=f"ai/logs/dqn_{role.__name__.lower()}.csv")
        except Exception as e:
            print(f"!!! {role.__name__} FAILED: {e} !!!")

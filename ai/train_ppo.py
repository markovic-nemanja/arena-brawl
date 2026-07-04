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
from ai.rollout_buffer import RolloutBuffer
from ai.ppo_agent import PPOAgent

ALL_ROLES = [Gunner, Bomber, Dasher, ToxicTrail, Blackhole]

PHASE_BOTS = {
    1: StationaryBot,
    2: RandomBot,
    3: RandomShooterBot,
    4: partial(AimShooterBot, aim_prob=0.5),
    5: partial(GentleAggressor, fire_prob=0.5),
    6: partial(GentleMedium, fire_prob=0.5),
}
GRADUATION_PHASE = max(PHASE_BOTS) + 1
GRAD_BOTS = [PHASE_BOTS[k] for k in (4, 5, 6)]
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

def train_ppo(agent_role=Gunner, total_steps=2_300_000, rollout_size=2048,
              save_dir="ai/weights", log_path="ai/logs/ppo_training.csv"):

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    env = ArenaBrawlEnv(agent_role=agent_role())
    agent = PPOAgent()
    buffer = RolloutBuffer()

    wandb.init(project="arena-brawl-ppo", name=agent_role.__name__,
               config={"total_steps": total_steps, "rollout_size": rollout_size,
                       "entropy_coef": agent.entropy_coef, "phase_promote": PHASE_PROMOTE},
               reinit=True)

    log_file = open(log_path, mode='w', newline="")
    writer = csv.writer(log_file)
    writer.writerow(["update", "steps", "phase", "avg_reward", "win_rate",
                     "entropy", "policy_loss", "value_loss", "episodes"])

    def set_opponent(phase):
        env.opponent_role = random.choice(ALL_ROLES)()   # random body -> all hazard types appear
        env.opponent_bot  = random.choice(GRAD_BOTS) if phase == GRADUATION_PHASE else PHASE_BOTS[phase]

    phase = 1
    phase_start_steps = 0
    finished = False
    recent_wins = deque(maxlen=WIN_WINDOW)
    set_opponent(phase)
    state, _ = env.reset()

    episode_reward = 0
    completed_rewards = []
    wins = 0
    steps_done = 0
    update_num = 0
    checkpoint_every = 500_000
    last_checkpoint = 0
    start_time = last_print = time.time()

    print(f"=== PPO {agent_role.__name__} | Phase {phase} ({_bot_name(phase)}) ===")

    while steps_done < total_steps and not finished:
        # Collect a rollout of transitions
        for _ in range(rollout_size):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.store(state, action, reward, done, log_prob, value)
            episode_reward += reward
            state = next_state
            steps_done += 1

            if done:
                completed_rewards.append(episode_reward)
                win = 1 if (terminated and env.agent.hp > 0) else 0# alive at termination == we won
                wins += win
                recent_wins.append(win)
                episode_reward = 0

                # Promote on win-rate mastery once past the floor; force-promote at the ceiling.
                if phase < MAX_PHASE:
                    in_phase = steps_done - phase_start_steps
                    wr = sum(recent_wins) / len(recent_wins) if recent_wins else 0
                    mastered = in_phase >= PHASE_MIN[phase] and len(recent_wins) >= WIN_WINDOW and wr >= PHASE_PROMOTE[phase]
                    if mastered or in_phase >= PHASE_MAX[phase]:
                        phase += 1
                        phase_start_steps = steps_done
                        recent_wins.clear()
                        why = "mastered" if mastered else "cap"
                        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                        print(f"[{elapsed}] === {agent_role.__name__} | Phase {phase} ({_bot_name(phase)}) @ step {steps_done} ({why}) ===")
                elif steps_done - phase_start_steps >= GRAD_STEPS:
                    finished = True   # graduation done — end the run (fixed 500k, no leftover accumulation)
                set_opponent(phase)
                
                state, _ = env.reset()

            if steps_done >= total_steps or finished:
                break

        # Update the policy using the collected rollout
        last_value = agent.select_action(state)[2]
        policy_loss, value_loss, entropy = agent.update(buffer, last_value)
        buffer.clear()
        update_num += 1

        if steps_done // checkpoint_every > last_checkpoint:
            last_checkpoint = steps_done // checkpoint_every
            agent.save(os.path.join(save_dir, f"ppo_{agent_role.__name__.lower()}_{steps_done}.pth"))

        avg_reward = sum(completed_rewards) / len(completed_rewards) if completed_rewards else 0
        win_rate = wins / len(completed_rewards) if completed_rewards else 0

        writer.writerow([update_num, steps_done, phase, round(avg_reward, 3), round(win_rate, 3),
                         round(entropy, 4), round(policy_loss, 4), round(value_loss, 4), len(completed_rewards)])
        log_file.flush()
        wandb.log({"update": update_num, "steps": steps_done, "phase": phase,
                   "avg_reward": avg_reward, "win_rate": win_rate, "entropy": entropy,
                   "policy_loss": policy_loss, "value_loss": value_loss, "episodes": len(completed_rewards)})

        if time.time() - last_print >= 30:
            elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            print(f"[{elapsed}] {agent_role.__name__} | update {update_num} | Phase{phase} ({_bot_name(phase)}) "
                  f"| steps {steps_done} | avg_reward {avg_reward:.2f} | win_rate {win_rate:.2f} "
                  f"| entropy {entropy:.3f} | episodes {len(completed_rewards)}")
            last_print = time.time()

        completed_rewards = []
        wins = 0

    agent.save(os.path.join(save_dir, f"ppo_{agent_role.__name__.lower()}_final.pth"))
    log_file.close()
    env.close()
    wandb.finish()
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"[{elapsed}] PPO {agent_role.__name__} DONE ({steps_done} steps).")

if __name__ == "__main__":
    for role in ALL_ROLES:
        print(f"\n######## PPO training: {role.__name__} ########")
        try:
            train_ppo(agent_role=role, total_steps=2_300_000,
                      log_path=f"ai/logs/ppo_{role.__name__.lower()}.csv")
        except Exception as e:
            print(f"!!! {role.__name__} FAILED: {e} !!!")

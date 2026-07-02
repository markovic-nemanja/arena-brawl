import os
import csv
import time
import random
from functools import partial

import wandb
from arena_env import ArenaBrawlEnv
from systems.roles import *
from systems.controller import (StationaryBot, RandomBot, RandomShooterBot,
                                 GentleAggressor, EasyBot, MediumBot, HardBot)
from ai.rollout_buffer import RolloutBuffer
from ai.ppo_agent import PPOAgent

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

def train_ppo(agent_role=Gunner, total_steps=2_000_000, rollout_size=2048,
              save_dir="ai/weights", log_path="ai/logs/ppo_training.csv"):

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    env = ArenaBrawlEnv(agent_role=agent_role())
    agent = PPOAgent()
    buffer = RolloutBuffer()

    wandb.init(project="arena-brawl-ppo", name=agent_role.__name__,
               config={"total_steps": total_steps, "rollout_size": rollout_size,
                       "entropy_coef": agent.entropy_coef, "phase_bounds": PHASE_BOUNDS},
               reinit=True)

    log_file = open(log_path, mode='w', newline="")
    writer = csv.writer(log_file)
    writer.writerow(["update", "steps", "phase", "avg_reward", "win_rate",
                     "entropy", "policy_loss", "value_loss", "episodes"])

    def set_opponent(phase):
        env.opponent_role = random.choice(ALL_ROLES)()   # random body -> all hazard types appear
        env.opponent_bot  = PHASE_BOTS[phase]

    last_phase = get_phase(0)
    set_opponent(last_phase)
    state, _ = env.reset()

    episode_reward = 0
    completed_rewards = []
    wins = 0
    steps_done = 0
    update_num = 0
    checkpoint_every = 500_000
    last_checkpoint = 0
    start_time = last_print = time.time()

    print(f"=== PPO {agent_role.__name__} | Phase {last_phase} ({_bot_name(last_phase)}) ===")

    while steps_done < total_steps:
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
                if terminated and env.opponent.hp <= 0:
                    wins += 1
                episode_reward = 0

                phase = get_phase(steps_done)
                if phase != last_phase:
                    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                    print(f"[{elapsed}] === {agent_role.__name__} | Phase {phase} ({_bot_name(phase)}) @ step {steps_done} ===")
                    last_phase = phase
                set_opponent(phase)
                state, _ = env.reset()

            if steps_done >= total_steps:
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
        phase = get_phase(steps_done)

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
            train_ppo(agent_role=role, total_steps=2_000_000,
                      log_path=f"ai/logs/ppo_{role.__name__.lower()}.csv")
        except Exception as e:
            print(f"!!! {role.__name__} FAILED: {e} !!!")

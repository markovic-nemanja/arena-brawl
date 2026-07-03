import argparse
import csv
import os
import random
import time
from functools import partial

import numpy as np
import torch
import wandb

from ai.ppo_agent import PPOAgent
from ai.rollout_buffer import VectorRolloutBuffer
from arena_env import create_vector_env, recommended_num_envs
from systems.controller import (
    EasyBot,
    GentleAggressor,
    MediumBot,
    RandomBot,
    RandomShooterBot,
    StationaryBot,
)
from systems.roles import Blackhole, Bomber, Dasher, Gunner, ToxicTrail


ALL_ROLES = [Gunner, Bomber, Dasher, ToxicTrail, Blackhole]
ROLE_BY_NAME = {role.__name__.lower(): role for role in ALL_ROLES}

PHASE_MIXES = {
    1: ((StationaryBot, 1.00),),
    2: ((StationaryBot, 0.25), (RandomBot, 0.75)),
    3: ((RandomBot, 0.25), (RandomShooterBot, 0.75)),
    4: (
        (RandomShooterBot, 0.25),
        (partial(GentleAggressor, fire_prob=0.4), 0.75),
    ),
    5: (
        (partial(GentleAggressor, fire_prob=0.4), 0.25),
        (partial(GentleAggressor, fire_prob=0.7), 0.75),
    ),
    6: (
        (partial(GentleAggressor, fire_prob=0.7), 0.25),
        (EasyBot, 0.75),
    ),
    7: ((EasyBot, 0.30), (MediumBot, 0.70)),
}
PHASE_BOUNDS = (250_000, 500_000, 800_000, 1_100_000, 1_400_000, 1_700_000)


def get_phase(steps):
    phase = 1
    for boundary in PHASE_BOUNDS:
        if steps < boundary:
            break
        phase += 1
    return phase


def _phase_end(phase, total_steps):
    if phase <= len(PHASE_BOUNDS):
        return min(PHASE_BOUNDS[phase - 1], total_steps)
    return total_steps


def _validate_parallel_budgets(num_envs, total_steps, rollout_size):
    values = [total_steps, rollout_size]
    values.extend(boundary for boundary in PHASE_BOUNDS if boundary < total_steps)
    invalid = [value for value in values if value % num_envs]
    if invalid:
        raise ValueError(
            f"num_envs={num_envs} must divide rollout_size, total_steps, and "
            f"phase boundaries; incompatible values: {invalid}"
        )


def _completed_wins(terminations, dones, infos):
    final_info = infos.get("final_info", {})
    opponent_hp = final_info.get("opponent_hp")
    if opponent_hp is None:
        return 0
    return sum(
        1
        for index in np.flatnonzero(dones)
        if terminations[index] and opponent_hp[index] <= 0
    )


def train_ppo(
    agent_role=Gunner,
    total_steps=2_000_000,
    rollout_size=2048,
    num_envs=None,
    asynchronous=True,
    save_dir="ai/weights",
    log_path="ai/logs/ppo_training.csv",
    seed=0,
):
    num_envs = num_envs or recommended_num_envs()
    _validate_parallel_budgets(num_envs, total_steps, rollout_size)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    agent = PPOAgent(state_size=20, action_size=10)
    buffer = VectorRolloutBuffer()
    wandb.init(
        project="arena-brawl-ppo",
        name=agent_role.__name__,
        config={
            "total_steps": total_steps,
            "rollout_size": rollout_size,
            "num_envs": num_envs,
            "phase_bounds": PHASE_BOUNDS,
            "seed": seed,
        },
        reinit=True,
    )
    log_file = open(log_path, mode="w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(
        [
            "update",
            "steps",
            "phase",
            "avg_reward",
            "win_rate",
            "entropy",
            "policy_loss",
            "value_loss",
            "episodes",
            "num_envs",
        ]
    )

    vector_env = None
    states = None
    episode_rewards = np.zeros(num_envs, dtype=np.float64)
    completed_rewards = []
    wins = 0
    steps_done = 0
    update_number = 0
    last_checkpoint = 0
    active_phase = None
    start_time = last_print = time.time()

    try:
        print(
            f"PPO {agent_role.__name__}: {num_envs} parallel environments, "
            f"device={agent.device}"
        )
        while steps_done < total_steps:
            phase = get_phase(steps_done)
            if phase != active_phase:
                if vector_env is not None:
                    vector_env.close()
                vector_env, states = create_vector_env(
                    agent_role,
                    PHASE_MIXES[phase],
                    ALL_ROLES,
                    num_envs,
                    seed + phase * 10_000,
                    asynchronous=asynchronous,
                )
                episode_rewards.fill(0.0)
                active_phase = phase
                print(f"PPO {agent_role.__name__}: entering phase {phase}")

            transitions = min(
                rollout_size,
                total_steps - steps_done,
                _phase_end(phase, total_steps) - steps_done,
            )
            vector_steps = transitions // num_envs
            buffer.clear()

            for _ in range(vector_steps):
                actions, log_probabilities, values = agent.select_actions(states)
                next_states, rewards, terminations, truncations, infos = (
                    vector_env.step(actions)
                )
                dones = np.logical_or(terminations, truncations)
                buffer.store(
                    states,
                    actions,
                    rewards,
                    dones,
                    log_probabilities,
                    values,
                )
                episode_rewards += rewards
                wins += _completed_wins(terminations, dones, infos)
                for index in np.flatnonzero(dones):
                    completed_rewards.append(float(episode_rewards[index]))
                    episode_rewards[index] = 0.0
                states = next_states
                steps_done += num_envs

            last_values = agent.get_values(states)
            policy_loss, value_loss, entropy = agent.update(buffer, last_values)
            update_number += 1

            if steps_done // 500_000 > last_checkpoint:
                last_checkpoint = steps_done // 500_000
                agent.save(
                    os.path.join(
                        save_dir,
                        f"ppo_{agent_role.__name__.lower()}_{steps_done}.pth",
                    )
                )

            average_reward = (
                sum(completed_rewards) / len(completed_rewards)
                if completed_rewards
                else 0.0
            )
            win_rate = wins / len(completed_rewards) if completed_rewards else 0.0
            writer.writerow(
                [
                    update_number,
                    steps_done,
                    phase,
                    round(average_reward, 4),
                    round(win_rate, 4),
                    round(entropy, 4),
                    round(policy_loss, 4),
                    round(value_loss, 4),
                    len(completed_rewards),
                    num_envs,
                ]
            )
            log_file.flush()
            wandb.log(
                {
                    "update": update_number,
                    "steps": steps_done,
                    "phase": phase,
                    "avg_reward": average_reward,
                    "win_rate": win_rate,
                    "entropy": entropy,
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "num_envs": num_envs,
                    "steps_per_second": steps_done
                    / max(time.time() - start_time, 1e-6),
                }
            )

            if time.time() - last_print >= 30:
                elapsed = time.strftime(
                    "%H:%M:%S", time.gmtime(time.time() - start_time)
                )
                print(
                    f"[{elapsed}] PPO {agent_role.__name__} | phase {phase} "
                    f"| steps {steps_done} | reward {average_reward:.3f} "
                    f"| win_rate {win_rate:.2f} | entropy {entropy:.3f} "
                    f"| SPS {steps_done / max(time.time() - start_time, 1e-6):.0f}"
                )
                last_print = time.time()
            completed_rewards.clear()
            wins = 0

        agent.save(
            os.path.join(
                save_dir, f"ppo_{agent_role.__name__.lower()}_final.pth"
            )
        )
    finally:
        if vector_env is not None:
            vector_env.close()
        log_file.close()
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO in parallel arenas.")
    parser.add_argument("--role", choices=ROLE_BY_NAME, default="gunner")
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--rollout-size", type=int, default=2048)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    role = ROLE_BY_NAME[args.role]
    train_ppo(
        agent_role=role,
        total_steps=args.total_steps,
        rollout_size=args.rollout_size,
        num_envs=args.num_envs,
        seed=args.seed,
        log_path=f"ai/logs/ppo_{args.role}.csv",
    )

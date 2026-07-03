import argparse
import csv
import os
import random
import time
from collections import deque
from functools import partial

import numpy as np
import torch
import wandb

from ai.dqn_agent import DQNAgent
from ai.replay_buffer import ReplayBuffer
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


def _validate_parallel_budgets(num_envs, total_steps):
    values = [total_steps]
    values.extend(boundary for boundary in PHASE_BOUNDS if boundary < total_steps)
    invalid = [value for value in values if value % num_envs]
    if invalid:
        raise ValueError(
            f"num_envs={num_envs} must divide total_steps and phase "
            f"boundaries; incompatible values: {invalid}"
        )


def _completed_wins(terminations, dones, infos):
    final_info = infos.get("final_info", {})
    opponent_hp = final_info.get("opponent_hp")
    if opponent_hp is None:
        return np.zeros(len(dones), dtype=np.int64)
    return np.asarray(
        [
            int(dones[index] and terminations[index] and opponent_hp[index] <= 0)
            for index in range(len(dones))
        ],
        dtype=np.int64,
    )


def train_dqn(
    agent_role=Gunner,
    total_steps=2_000_000,
    buffer_capacity=100_000,
    batch_size=64,
    num_envs=None,
    asynchronous=True,
    save_dir="ai/weights",
    log_path="ai/logs/dqn_training.csv",
    seed=0,
    agent_class=DQNAgent,
    algorithm_name="dqn",
):
    num_envs = num_envs or recommended_num_envs()
    _validate_parallel_budgets(num_envs, total_steps)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    agent = agent_class(state_size=20, action_size=10, batch_size=batch_size)
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)
    wandb.init(
        project=f"arena-brawl-{algorithm_name}",
        name=agent_role.__name__,
        config={
            "total_steps": total_steps,
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
            "episodes",
            "steps",
            "phase",
            "avg_reward",
            "win_rate",
            "epsilon",
            "loss",
            "num_envs",
        ]
    )

    vector_env = None
    states = None
    episode_rewards = np.zeros(num_envs, dtype=np.float64)
    recent_wins = deque(maxlen=100)
    steps_done = 0
    episodes = 0
    last_checkpoint = 0
    active_phase = None
    start_time = last_print = time.time()

    try:
        print(
            f"{algorithm_name.upper()} {agent_role.__name__}: {num_envs} "
            f"parallel environments, device={agent.device}"
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
                if active_phase is not None:
                    agent.epsilon = max(agent.epsilon, 0.5)
                active_phase = phase
                print(
                    f"{algorithm_name.upper()} {agent_role.__name__}: "
                    f"entering phase {phase}"
                )

            phase_end = _phase_end(phase, total_steps)
            completed_this_phase = []
            total_loss = 0.0
            loss_count = 0

            while steps_done < phase_end:
                actions = agent.select_actions(states)
                next_states, rewards, terminations, truncations, infos = (
                    vector_env.step(actions)
                )
                dones = np.logical_or(terminations, truncations)
                previous_buffer_size = len(replay_buffer)
                replay_buffer.store_batch(
                    states, actions, rewards, next_states, dones.astype(np.float32)
                )

                # Preserve the old DQN update-to-transition ratio: collecting N
                # transitions performs N optimizer updates.
                gradient_steps = min(
                    num_envs,
                    max(0, previous_buffer_size + num_envs - batch_size + 1),
                )
                for _ in range(gradient_steps):
                    loss = agent.train_step(replay_buffer)
                    if loss is not None:
                        total_loss += loss
                        loss_count += 1

                episode_rewards += rewards
                wins = _completed_wins(terminations, dones, infos)
                for index in np.flatnonzero(dones):
                    completed_this_phase.append(float(episode_rewards[index]))
                    recent_wins.append(int(wins[index]))
                    episode_rewards[index] = 0.0
                    episodes += 1
                    agent.decay_epsilon()

                states = next_states
                steps_done += num_envs

                if steps_done // 500_000 > last_checkpoint:
                    last_checkpoint = steps_done // 500_000
                    agent.save(
                        os.path.join(
                            save_dir,
                            f"{algorithm_name}_{agent_role.__name__.lower()}_"
                            f"{steps_done}.pth",
                        )
                    )

                if time.time() - last_print >= 30:
                    average_reward = (
                        sum(completed_this_phase) / len(completed_this_phase)
                        if completed_this_phase
                        else 0.0
                    )
                    win_rate = (
                        sum(recent_wins) / len(recent_wins) if recent_wins else 0.0
                    )
                    average_loss = total_loss / loss_count if loss_count else 0.0
                    elapsed = time.strftime(
                        "%H:%M:%S", time.gmtime(time.time() - start_time)
                    )
                    print(
                        f"[{elapsed}] {algorithm_name.upper()} "
                        f"{agent_role.__name__} | phase {phase} "
                        f"| steps {steps_done} | reward {average_reward:.3f} "
                        f"| win_rate {win_rate:.2f} | epsilon {agent.epsilon:.3f} "
                        f"| SPS {steps_done / max(time.time() - start_time, 1e-6):.0f}"
                    )
                    last_print = time.time()

            average_reward = (
                sum(completed_this_phase) / len(completed_this_phase)
                if completed_this_phase
                else 0.0
            )
            win_rate = sum(recent_wins) / len(recent_wins) if recent_wins else 0.0
            average_loss = total_loss / loss_count if loss_count else 0.0
            writer.writerow(
                [
                    episodes,
                    steps_done,
                    phase,
                    round(average_reward, 4),
                    round(win_rate, 4),
                    round(agent.epsilon, 4),
                    round(average_loss, 6),
                    num_envs,
                ]
            )
            log_file.flush()
            wandb.log(
                {
                    "episodes": episodes,
                    "steps": steps_done,
                    "phase": phase,
                    "avg_reward": average_reward,
                    "win_rate": win_rate,
                    "epsilon": agent.epsilon,
                    "loss": average_loss,
                    "num_envs": num_envs,
                    "steps_per_second": steps_done
                    / max(time.time() - start_time, 1e-6),
                }
            )

        agent.save(
            os.path.join(
                save_dir,
                f"{algorithm_name}_{agent_role.__name__.lower()}_final.pth",
            )
        )
    finally:
        if vector_env is not None:
            vector_env.close()
        log_file.close()
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN in parallel arenas.")
    parser.add_argument("--role", choices=ROLE_BY_NAME, default="gunner")
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--buffer-capacity", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    role = ROLE_BY_NAME[args.role]
    train_dqn(
        agent_role=role,
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        seed=args.seed,
        log_path=f"ai/logs/dqn_{args.role}.csv",
    )

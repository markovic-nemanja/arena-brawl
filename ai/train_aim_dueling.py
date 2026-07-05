"""Parallel aim training for Dueling DQN.

Independent CPU workers simulate arenas while the parent process owns the
network, batches GPU inference, trains from one replay buffer, and saves models.
"""

import argparse
import csv
import os
import time
from functools import partial

import numpy as np
import torch
import wandb
from gymnasium.vector import AsyncVectorEnv, AutoresetMode

from ai.dueling_dqn_agent import DuelingDQNAgent
from ai.replay_buffer import ReplayBuffer
from arena_env import ArenaBrawlEnv
from systems.controller import StationaryBot
from systems.roles import Blackhole, Bomber, Dasher, Gunner, ToxicTrail


EPS_END = 0.05
ROLES = {
    "gunner": Gunner,
    "bomber": Bomber,
    "dasher": Dasher,
    "toxictrail": ToxicTrail,
    "blackhole": Blackhole,
}


def create_aim_env(role_name, opponent_bot):
    """Top-level worker factory so multiprocessing can pickle it."""
    role_class = ROLES[role_name]
    return ArenaBrawlEnv(
        agent_role=role_class(),
        opponent_role=role_class(),
        opponent_bot=opponent_bot,
        aim_practice=True,
        render_mode=None,
    )


def train_aim_dueling(
    agent_role=Gunner,
    total_steps=1_000_000,
    opponent_bot=StationaryBot,
    base_weights=None,
    eps_start=1.0,
    eps_anneal_frac=0.6,
    num_envs=16,
    buffer_capacity=200_000,
    batch_size=256,
    gradient_steps=4,
    warmup_steps=10_000,
    checkpoint_every=100_000,
    log_every=10_000,
    save_dir="ai/weights",
    log_path="ai/logs/dueling_aim.csv",
    save_suffix="aim",
    wandb_project="arena-brawl-dueling-aim",
    wandb_mode="online",
    seed=1234,
):
    if num_envs < 1:
        raise ValueError("num_envs must be at least 1")
    if gradient_steps < 1:
        raise ValueError("gradient_steps must be at least 1")

    os.makedirs(save_dir, exist_ok=True)
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    torch.set_num_threads(1)

    role_name = agent_role.__name__.lower()
    envs = AsyncVectorEnv(
        [
            partial(create_aim_env, role_name, opponent_bot)
            for _ in range(num_envs)
        ],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )

    agent = DuelingDQNAgent(state_size=22, batch_size=batch_size)
    if base_weights:
        agent.load(base_weights)
        print(f"[transfer] loaded {base_weights}")

    buffer = ReplayBuffer(capacity=buffer_capacity)
    anneal_steps = max(1, int(total_steps * eps_anneal_frac))
    effective_warmup = max(batch_size, warmup_steps)

    run = wandb.init(
        project=wandb_project,
        name=f"dueling-aim-{role_name}-{num_envs}envs",
        group="dueling-aim-all-roles",
        mode=wandb_mode,
        config={
            "algorithm": "dueling_dqn",
            "role": agent_role.__name__,
            "opponent": opponent_bot.__name__,
            "num_envs": num_envs,
            "total_steps": total_steps,
            "batch_size": batch_size,
            "gradient_steps": gradient_steps,
            "warmup_steps": effective_warmup,
            "buffer_capacity": buffer_capacity,
            "eps_start": eps_start,
            "eps_end": EPS_END,
            "eps_anneal_frac": eps_anneal_frac,
        },
        reinit="finish_previous",
    )

    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(
        [
            "steps",
            "episodes",
            "avg_reward",
            "epsilon",
            "loss",
            "steps_per_second",
            "buffer_size",
        ]
    )

    steps_done = 0
    episodes = 0
    next_checkpoint = checkpoint_every if checkpoint_every > 0 else None
    last_log_step = 0
    episode_rewards = np.zeros(num_envs, dtype=np.float32)
    completed_rewards = []
    recent_losses = []
    start_time = time.time()

    print(
        f"=== AIM-DUELING {agent_role.__name__} vs {opponent_bot.__name__} "
        f"| {num_envs} envs | device={agent.device} ==="
    )

    try:
        states, _ = envs.reset(seed=seed)

        while steps_done < total_steps:
            progress = min(steps_done / anneal_steps, 1.0)
            agent.epsilon = max(
                EPS_END,
                eps_start - (eps_start - EPS_END) * progress,
            )

            actions = agent.select_actions(states)
            next_states, rewards, terminated, truncated, _ = envs.step(actions)
            dones = np.logical_or(terminated, truncated)

            buffer.store_batch(
                states,
                actions,
                rewards,
                next_states,
                dones.astype(np.float32),
            )

            episode_rewards += rewards
            states = next_states
            steps_done += num_envs

            if len(buffer) >= effective_warmup:
                for _ in range(gradient_steps):
                    loss = agent.train_step(buffer)
                    if loss is not None:
                        recent_losses.append(loss)

            for index in np.flatnonzero(dones):
                completed_rewards.append(float(episode_rewards[index]))
                episode_rewards[index] = 0.0
                episodes += 1

            while next_checkpoint is not None and steps_done >= next_checkpoint:
                checkpoint_path = os.path.join(
                    save_dir,
                    f"dueling_{role_name}_{save_suffix}_{next_checkpoint}.pth",
                )
                agent.save(checkpoint_path)
                next_checkpoint += checkpoint_every

            if steps_done - last_log_step >= log_every or steps_done >= total_steps:
                elapsed = max(time.time() - start_time, 1e-6)
                steps_per_second = steps_done / elapsed
                average_reward = (
                    float(np.mean(completed_rewards))
                    if completed_rewards
                    else None
                )
                average_loss = (
                    float(np.mean(recent_losses)) if recent_losses else 0.0
                )

                writer.writerow(
                    [
                        steps_done,
                        episodes,
                        round(average_reward, 4)
                        if average_reward is not None
                        else "",
                        round(agent.epsilon, 5),
                        round(average_loss, 6),
                        round(steps_per_second, 2),
                        len(buffer),
                    ]
                )
                log_file.flush()

                metrics = {
                    "steps": steps_done,
                    "episodes": episodes,
                    "epsilon": agent.epsilon,
                    "loss": average_loss,
                    "steps_per_second": steps_per_second,
                    "buffer_size": len(buffer),
                }
                if average_reward is not None:
                    metrics["avg_reward"] = average_reward
                run.log(metrics)

                reward_text = (
                    f"{average_reward:.1f}"
                    if average_reward is not None
                    else "n/a"
                )
                print(
                    f"{role_name} | steps {steps_done} | episodes {episodes} "
                    f"| reward {reward_text} | eps {agent.epsilon:.3f} "
                    f"| loss {average_loss:.4f} | {steps_per_second:.0f} steps/s"
                )

                completed_rewards.clear()
                recent_losses.clear()
                last_log_step = steps_done

        final_path = os.path.join(
            save_dir,
            f"dueling_{role_name}_{save_suffix}.pth",
        )
        agent.save(final_path)
        print(f"Saved {final_path}")
    finally:
        log_file.close()
        envs.close()
        run.finish()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel Dueling DQN aim training.",
    )
    parser.add_argument("--role", choices=["all", *ROLES], default="all")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gradient-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=10_000)
    parser.add_argument("--buffer-capacity", type=int, default=200_000)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-anneal-frac", type=float, default=0.6)
    parser.add_argument("--checkpoint-every", type=int, default=100_000)
    parser.add_argument("--log-every", type=int, default=10_000)
    parser.add_argument("--save-dir", default="ai/weights")
    parser.add_argument("--log-dir", default="ai/logs")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    selected_roles = list(ROLES) if args.role == "all" else [args.role]

    for role_name in selected_roles:
        role_class = ROLES[role_name]
        train_aim_dueling(
            agent_role=role_class,
            total_steps=args.total_steps,
            num_envs=args.num_envs,
            batch_size=args.batch_size,
            gradient_steps=args.gradient_steps,
            warmup_steps=args.warmup_steps,
            buffer_capacity=args.buffer_capacity,
            eps_start=args.eps_start,
            eps_anneal_frac=args.eps_anneal_frac,
            checkpoint_every=args.checkpoint_every,
            log_every=args.log_every,
            save_dir=args.save_dir,
            log_path=os.path.join(
                args.log_dir,
                f"dueling_{role_name}_aim.csv",
            ),
            wandb_mode=args.wandb_mode,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()

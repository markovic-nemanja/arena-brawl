import argparse
import csv
import os
import random
import time
from collections import deque
from functools import partial

import wandb

from arena_env import ArenaBrawlEnv
from ai.dueling_dqn_agent import DuelingDQNAgent
from ai.replay_buffer import ReplayBuffer
from systems.controller import (
    EasyBot,
    GentleAggressor,
    MediumBot,
    RandomBot,
    RandomShooterBot,
    StationaryBot,
)
from systems.roles import Blackhole, Bomber, Dasher, Gunner, ToxicTrail


OPPONENT_ROLES = [Gunner, Bomber, Dasher, ToxicTrail, Blackhole]

PHASE_BOTS = {
    1: StationaryBot,
    2: RandomBot,
    3: RandomShooterBot,
    4: partial(GentleAggressor, fire_prob=0.4),
    5: partial(GentleAggressor, fire_prob=0.7),
    6: EasyBot,
    7: MediumBot,
}

PHASE_BOUNDS = (
    200_000,
    400_000,
    650_000,
    950_000,
    1_250_000,
    1_600_000,
)


def get_phase(steps):
    phase = 1
    for boundary in PHASE_BOUNDS:
        if steps < boundary:
            break
        phase += 1
    return phase


def get_bot_name(phase):
    bot = PHASE_BOTS[phase]
    return bot.func.__name__ if isinstance(bot, partial) else bot.__name__


def train_dueling_dqn(
    total_steps=2_000_000,
    buffer_capacity=100_000,
    batch_size=64,
    save_dir="ai/weights",
    log_path="ai/logs/dueling_dqn_gunner.csv",
    checkpoint_every=500_000,
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    env = ArenaBrawlEnv(agent_role=Gunner())
    agent = DuelingDQNAgent(batch_size=batch_size)
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    wandb.init(
        project="arena-brawl-dueling-dqn",
        name="Gunner",
        config={
            "total_steps": total_steps,
            "batch_size": batch_size,
            "buffer_capacity": buffer_capacity,
            "gamma": agent.gamma,
            "epsilon_decay": agent.epsilon_decay,
            "phase_bounds": PHASE_BOUNDS,
        },
    )

    log_file = open(log_path, mode="w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(
        [
            "episode",
            "steps",
            "phase",
            "reward",
            "win",
            "epsilon",
            "loss",
            "episode_length",
        ]
    )

    def set_opponent(phase):
        env.opponent_role = random.choice(OPPONENT_ROLES)()
        env.opponent_bot = PHASE_BOTS[phase]

    last_phase = get_phase(0)
    set_opponent(last_phase)

    steps_done = 0
    episode = 0
    last_checkpoint = 0
    recent_wins = deque(maxlen=100)
    start_time = time.time()
    last_print = start_time

    print(
        f"=== Dueling DQN Gunner | Phase {last_phase} "
        f"({get_bot_name(last_phase)}) ==="
    )

    try:
        while steps_done < total_steps:
            state, _ = env.reset()
            episode_reward = 0.0
            total_loss = 0.0
            loss_count = 0
            episode_length = 0

            while True:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                replay_buffer.store(
                    state,
                    action,
                    reward,
                    next_state,
                    float(done),
                )

                loss = agent.train_step(replay_buffer)
                if loss is not None:
                    total_loss += loss
                    loss_count += 1

                state = next_state
                episode_reward += reward
                episode_length += 1
                steps_done += 1

                if done or steps_done >= total_steps:
                    break

            win = 1 if terminated and env.opponent.hp <= 0 else 0
            recent_wins.append(win)
            agent.decay_epsilon()
            episode += 1

            phase = get_phase(steps_done)
            if phase != last_phase:
                agent.epsilon = max(agent.epsilon, 0.5)
                elapsed = time.strftime(
                    "%H:%M:%S",
                    time.gmtime(time.time() - start_time),
                )
                print(
                    f"[{elapsed}] === Gunner | Phase {phase} "
                    f"({get_bot_name(phase)}) @ step {steps_done} ==="
                )
                last_phase = phase

            set_opponent(phase)

            average_loss = total_loss / loss_count if loss_count else 0.0
            win_rate = sum(recent_wins) / len(recent_wins)

            writer.writerow(
                [
                    episode,
                    steps_done,
                    phase,
                    round(episode_reward, 3),
                    win,
                    round(agent.epsilon, 4),
                    round(average_loss, 5),
                    episode_length,
                ]
            )
            log_file.flush()

            wandb.log(
                {
                    "episode": episode,
                    "steps": steps_done,
                    "phase": phase,
                    "reward": episode_reward,
                    "win": win,
                    "win_rate": win_rate,
                    "epsilon": agent.epsilon,
                    "loss": average_loss,
                    "episode_length": episode_length,
                }
            )

            if steps_done // checkpoint_every > last_checkpoint:
                last_checkpoint = steps_done // checkpoint_every
                checkpoint_path = os.path.join(
                    save_dir,
                    f"dueling_dqn_gunner_{steps_done}.pth",
                )
                agent.save(checkpoint_path)

            if time.time() - last_print >= 30:
                elapsed = time.strftime(
                    "%H:%M:%S",
                    time.gmtime(time.time() - start_time),
                )
                print(
                    f"[{elapsed}] Gunner | episode {episode} | Phase {phase} "
                    f"({get_bot_name(phase)}) | steps {steps_done} "
                    f"| reward {episode_reward:.1f} | win {win} "
                    f"| win_rate {win_rate:.2f} | epsilon {agent.epsilon:.3f} "
                    f"| loss {average_loss:.5f}"
                )
                last_print = time.time()

        final_path = os.path.join(
            save_dir,
            "dueling_dqn_gunner_final.pth",
        )
        agent.save(final_path)
        print(f"Dueling DQN Gunner saved to {final_path}")
    finally:
        log_file.close()
        env.close()
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Dueling DQN agent for the Gunner role."
    )
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--buffer-capacity", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=500_000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_dueling_dqn(
        total_steps=args.total_steps,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
    )

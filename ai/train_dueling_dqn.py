import argparse

from ai.dueling_dqn_agent import DuelingDQNAgent
from ai.train_dqn import train_dqn
from systems.roles import Gunner, ToxicTrail


def train_dueling_dqn(
    agent_role=Gunner,
    total_steps=2_000_000,
    buffer_capacity=100_000,
    batch_size=64,
    num_envs=None,
    asynchronous=True,
    save_dir="ai/weights",
    log_path="ai/logs/dueling_dqn_training.csv",
    seed=0,
):
    # The training loop and curriculum are exactly the same as standard DQN;
    # only the Q-network architecture changes.
    return train_dqn(
        agent_role=agent_role,
        total_steps=total_steps,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        num_envs=num_envs,
        asynchronous=asynchronous,
        save_dir=save_dir,
        log_path=log_path,
        seed=seed,
        agent_class=DuelingDQNAgent,
        algorithm_name="dueling_dqn",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train Dueling DQN.")
    parser.add_argument("--role", choices=("gunner", "toxictrail"), default="gunner")
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    role = Gunner if args.role == "gunner" else ToxicTrail
    train_dueling_dqn(
        agent_role=role,
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        seed=args.seed,
        log_path=f"ai/logs/dueling_dqn_{args.role}.csv",
    )

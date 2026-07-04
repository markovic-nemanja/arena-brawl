"""Deterministic held-out benchmark shared by every learning algorithm."""

import argparse

from functools import partial

from arena_env import ArenaBrawlEnv
from systems.controller import (
    EasyBot,
    GentleAggressor,
    HardBot,
    MediumBot,
    RandomBot,
    RandomShooterBot,
    StationaryBot,
)
from systems.roles import Bomber, Dasher, Gunner, ToxicTrail
from ai.ppo_agent import PPOAgent
from ai.dqn_agent import DQNAgent
from ai.dueling_dqn_agent import DuelingDQNAgent


ROLE_BY_NAME = {
    role.__name__.lower(): role
    for role in (Gunner, Bomber, Dasher, ToxicTrail)
}

TEST_BOTS = (
    ("stationary", StationaryBot),
    ("random", RandomBot),
    ("random_shooter", RandomShooterBot),
    ("gentle_50", partial(GentleAggressor, fire_prob=0.50)),
    ("easy", EasyBot),
    ("medium", MediumBot),
    ("hard", HardBot),
)


def _load(weights, algo):
    agents = {
        "dqn": DQNAgent,
        "dueling_dqn": DuelingDQNAgent,
        "ppo": PPOAgent,
    }
    if algo not in agents:
        raise ValueError(f"Unknown algorithm: {algo}")

    # Version-2 experiments use the same 31 observations and 18 actions for
    # every algorithm. Legacy checkpoints must be evaluated on legacy code.
    agent = agents[algo](state_size=31, action_size=18)
    agent.load(weights)
    return agent


def evaluate(
    weights,
    role,
    algo="ppo",
    bots=TEST_BOTS,
    n_games=100,
    seed=9_000_000,
):
    agent = _load(weights, algo)
    scores = {}
    for bot_index, (bot_name, bot_factory) in enumerate(bots):
        env = ArenaBrawlEnv(
            agent_role=role(),
            opponent_role=role(),
            opponent_bot=bot_factory,
        )
        wins = losses = timeouts = 0
        damage_dealt = damage_taken = 0.0
        wall_blocks = casts = decisions = 0
        aim_total = 0.0
        try:
            for game in range(n_games):
                state, _ = env.reset(
                    seed=seed + bot_index * 100_000 + game
                )
                done = False
                while not done:
                    state, _, terminated, truncated, info = env.step(
                        agent.act(state)
                    )
                    damage_dealt += info["damage_dealt"]
                    damage_taken += info["damage_taken"]
                    wall_blocks += int(info["wall_blocked"])
                    casts += int(info["ability_cast"])
                    aim_total += float(info["aim_alignment"])
                    decisions += 1
                    done = terminated or truncated
                wins += int(terminated and env.opponent.hp <= 0)
                losses += int(terminated and env.agent.hp <= 0)
                timeouts += int(truncated)
        finally:
            env.close()

        scores[bot_name] = {
            "win_rate": wins / n_games,
            "loss_rate": losses / n_games,
            "timeout_rate": timeouts / n_games,
            "damage_dealt": damage_dealt / n_games,
            "damage_taken": damage_taken / n_games,
            "wall_block_rate": wall_blocks / max(decisions, 1),
            "casts_per_game": casts / n_games,
            "cast_aim": aim_total / max(casts, 1),
        }

    print(f"{algo.upper()} {role.__name__}")
    for name, metrics in scores.items():
        print(
            f"  {name:15s} win={metrics['win_rate']:.0%} "
            f"timeout={metrics['timeout_rate']:.0%} "
            f"damage={metrics['damage_dealt']:.1f}/"
            f"{metrics['damage_taken']:.1f} "
            f"aim={metrics['cast_aim']:.2f}"
        )
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=ROLE_BY_NAME, default="gunner")
    parser.add_argument(
        "--algo", choices=("ppo", "dqn", "dueling_dqn"), default="ppo"
    )
    parser.add_argument("--weights", default=None)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=9_000_000)
    args = parser.parse_args()
    weights = args.weights or (
        f"ai/weights/{args.algo}_{args.role}_v2_final.pth"
    )
    evaluate(
        weights,
        ROLE_BY_NAME[args.role],
        algo=args.algo,
        n_games=args.games,
        seed=args.seed,
    )

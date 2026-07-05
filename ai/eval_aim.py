"""GREEDY EVALUATION of aim-trained agents — the fair DQN vs PPO vs Dueling comparison.

WHY THIS EXISTS (not just reading the training CSVs): the training logs are collected DURING learning,
so they are contaminated by exploration. DQN especially still has epsilon ~0.48 at the end of 1M steps
(it takes ~half random actions), while PPO samples from a fairly committed policy — so the raw CSV
rewards understate DQN and are not a fair head-to-head. Here every agent is run GREEDY (argmax, no
epsilon, no sampling), headless, on identical seeds, so the numbers reflect the learned POLICY only.

The metrics are RATES/RATIOS so they are comparable ACROSS roles (raw hit_reward is not — a Gunner hit
and a Bomber hit remove different hp):
  - accuracy  = kills / shot        -> targets destroyed per ability use (the "aim accuracy")
  - aim_pct   = of MOVE decisions, fraction that head TOWARD the target (state-dependence: did it learn
                to read the target's position, or is it a fixed-direction bot?)
  - kills/ep  = targets destroyed per fixed-length episode (throughput: aim + positioning)
  - fire_pct  = fraction of decisions that fire (exposes spam, e.g. Bomber)

kills come straight from the reward: every kill removes exactly PLAYER_MAX_HP and reward = 0.3*damage,
so kills = sum(reward) / 0.3 / PLAYER_MAX_HP.

Set STAGE below. Prints a table and writes ai/logs/eval_<stage>.csv.
Run:  python -m ai.eval_aim
"""
import os
import csv
import math
import random
import numpy as np

from arena_env import ArenaBrawlEnv
from systems.roles import Gunner, Bomber, Dasher, ToxicTrail, Blackhole
from systems.controller import StationaryBot, RandomBot, _ACTION_TO_MOVE
from ai.ppo_agent import PPOAgent
from ai.dqn_agent import DQNAgent
from ai.dueling_dqn_agent import DuelingDQNAgent
from ai.a2c_agent import A2CAgent
import settings as S

# ---- config ----
STAGE = "aim_move" # "aim" = stationary target (StationaryBot); "aim_move" = moving target (RandomBot)
N_EPISODES = 15 # greedy episodes averaged per (algo, role); more = less noise, slower
ROLES = [Gunner, Bomber, Dasher, ToxicTrail, Blackhole]
ALGOS = ["ppo", "dqn", "dueling", "a2c"]

# each algo -> build a fresh agent (Dueling still defaults to state_size=20, so force 22)
AGENTS = {
    "ppo": lambda: PPOAgent(),
    "dqn": lambda: DQNAgent(),
    "dueling": lambda: DuelingDQNAgent(state_size=22),
    "a2c": lambda: A2CAgent(state_size=22),
}
OPP_BOT = StationaryBot if STAGE == "aim" else RandomBot
_R_DAMAGE_DEALT = 0.3   # must match arena_env; kills = reward / this / PLAYER_MAX_HP


def eval_one(algo, role):
    """Run N_EPISODES greedy episodes; return per-episode-averaged metrics (or None if no weights)."""
    weights = f"ai/weights/{algo}_{role.__name__.lower()}_{STAGE}.pth"
    if not os.path.exists(weights):
        return None
    agent = AGENTS[algo]()
    agent.load(weights)

    env = ArenaBrawlEnv(agent_role=role(), opponent_role=role(),
                        opponent_bot=OPP_BOT, aim_practice=True)

    tot_reward = tot_shots = tot_moves = tot_aimed = tot_decisions = 0.0
    for ep in range(N_EPISODES):
        random.seed(ep); np.random.seed(ep) # same target stream for every algo -> fair, reproducible
        state, _ = env.reset(seed=ep)
        done = False
        while not done:
            # measure at DECISION time (before the env advances)
            cd_ready = env.agent.cooldown <= 0
            dx = env.opponent.x - env.agent.x
            dy = env.opponent.y - env.agent.y
            dist = math.hypot(dx, dy)

            action = agent.act(state) # GREEDY
            tot_decisions += 1
            if action == 9 and cd_ready:
                tot_shots += 1
            elif 1 <= action <= 8:
                tot_moves += 1
                mvx, mvy = _ACTION_TO_MOVE[action]
                if dist > 0 and (mvx * dx + mvy * dy) / dist > 0: # heading toward the target
                    tot_aimed += 1

            state, reward, terminated, truncated, _ = env.step(action)
            tot_reward += reward
            done = terminated or truncated
    env.close()

    kills = tot_reward / _R_DAMAGE_DEALT / S.PLAYER_MAX_HP
    return {
        "kills_ep":  kills / N_EPISODES,
        "shots_ep":  tot_shots / N_EPISODES,
        "accuracy":  kills / tot_shots if tot_shots else 0.0, # kills per effective shot
        "aim_pct":   tot_aimed / tot_moves if tot_moves else 0.0,
        "fire_pct":  tot_shots / tot_decisions if tot_decisions else 0.0,
        "reward_ep": tot_reward / N_EPISODES,
    }


def main():
    os.makedirs("ai/logs", exist_ok=True)
    out_path = f"ai/logs/eval_{STAGE}.csv"
    rows = []
    print(f"=== GREEDY EVAL | stage={STAGE} | opponent={OPP_BOT.__name__} | {N_EPISODES} eps/role ===")
    header = f"{'algo':<8}{'role':<12}{'kills/ep':>9}{'shots/ep':>9}{'accuracy':>10}{'aim%':>8}{'fire%':>8}{'reward/ep':>11}"
    print(header)
    print("-" * len(header))
    for algo in ALGOS:
        for role in ROLES:
            m = eval_one(algo, role)
            if m is None:
                print(f"{algo:<8}{role.__name__:<12}{'  (no weights)':>20}")
                continue
            print(f"{algo:<8}{role.__name__:<12}{m['kills_ep']:>9.2f}{m['shots_ep']:>9.1f}"
                  f"{m['accuracy']:>10.3f}{m['aim_pct']*100:>7.1f}%{m['fire_pct']*100:>7.1f}%{m['reward_ep']:>11.1f}")
            rows.append([algo, role.__name__, round(m["kills_ep"], 3), round(m["shots_ep"], 2),
                         round(m["accuracy"], 4), round(m["aim_pct"], 4),
                         round(m["fire_pct"], 4), round(m["reward_ep"], 2)])

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algo", "role", "kills_ep", "shots_ep", "accuracy", "aim_pct", "fire_pct", "reward_ep"])
        w.writerows(rows)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

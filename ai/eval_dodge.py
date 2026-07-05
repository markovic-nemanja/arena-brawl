"""GREEDY EVALUATION of dodge-trained agents — the fair DQN vs PPO vs Dueling comparison for dodging.

Like eval_aim.py but for the dodge stage: every agent is run GREEDY (argmax, no epsilon/sampling),
headless, on identical seeds, against the same moving AimShooter(0.7) firing Gunner bullets. The
training CSVs are exploration-contaminated (DQN especially), so greedy eval is the only fair comparison.

The opponent and hazard are IDENTICAL for every role (always a Gunner shooter), and the agent is
immortal, so the metrics are comparable across roles AND algorithms:
  - dmg_taken/ep : total hp lost per fixed-length episode (LOWER = better). The headline dodge score.
                   Derived from the reward: reward = -_R_DAMAGE_TAKEN * damage, so damage = -reward / it.
  - wall%        : fraction of decisions spent hugging a wall (LOWER = better — walls deal damage, so a
                   good dodger stays in open space). This is the "did it learn to avoid walls" check.
  - move%        : fraction of decisions that are a MOVE (a frozen agent gets shot; dodging = moving).
  - reward/ep    : mean episode reward (<= 0), for continuity with the training logs.

Prints a table and writes ai/logs/eval_dodge.csv.
Run:  python -m ai.eval_dodge
"""
import os
import csv
import random
import numpy as np

from arena_env import ArenaBrawlEnv
from systems.roles import Gunner, Bomber, Dasher, ToxicTrail, Blackhole
from systems.controller import AimShooterBot
from ai.ppo_agent import PPOAgent
from ai.dqn_agent import DQNAgent
from ai.dueling_dqn_agent import DuelingDQNAgent
import settings as S

# ---- config ----
N_EPISODES = 15        # greedy episodes averaged per (algo, role); more = less noise, slower
ROLES = [Gunner, Bomber, Dasher, ToxicTrail, Blackhole]
ALGOS = ["ppo", "dqn"]   # focus on PPO vs DQN for now (Dueling/A2C are the friend's)

# each algo -> build a fresh agent (Dueling still defaults to state_size=20, so force 22)
AGENTS = {
    "ppo":     lambda: PPOAgent(),
    "dqn":     lambda: DQNAgent(),
    "dueling": lambda: DuelingDQNAgent(state_size=22),
}
_R_DAMAGE_TAKEN = 0.1   # must match arena_env; damage = -reward / this
_WALL_MARGIN = 70       # px from a wall counted as "hugging the wall"


def _near_wall(x, y):
    r = S.PLAYER_RADIUS
    return (x < S.ARENA_LEFT + r + _WALL_MARGIN or x > S.ARENA_RIGHT - r - _WALL_MARGIN or
            y < S.ARENA_TOP + r + _WALL_MARGIN or y > S.ARENA_BOTTOM - r - _WALL_MARGIN)


def eval_one(algo, role):
    """Run N_EPISODES greedy episodes; return per-episode-averaged metrics (or None if no weights)."""
    weights = f"ai/weights/{algo}_{role.__name__.lower()}_dodge.pth"
    shared = False
    if not os.path.exists(weights):
        # dodging is role-independent movement, so roles without their own dodge policy reuse the
        # shared Gunner dodge policy (only Dasher trains its own — it can dash to dodge).
        weights = f"ai/weights/{algo}_gunner_dodge.pth"
        shared = True
    if not os.path.exists(weights):
        return None
    agent = AGENTS[algo]()
    agent.load(weights)

    env = ArenaBrawlEnv(agent_role=role(), opponent_role=Gunner(),
                        opponent_bot=lambda: AimShooterBot(0.7), dodge_practice=True)

    tot_reward = tot_moves = tot_wall = tot_decisions = 0.0
    for ep in range(N_EPISODES):
        random.seed(ep); np.random.seed(ep)        # same shooter behaviour for every algo -> fair, reproducible
        state, _ = env.reset(seed=ep)
        done = False
        while not done:
            if _near_wall(env.agent.x, env.agent.y):
                tot_wall += 1
            action = agent.act(state)              # GREEDY
            tot_decisions += 1
            if 1 <= action <= 8:
                tot_moves += 1
            state, reward, terminated, truncated, _ = env.step(action)
            tot_reward += reward
            done = terminated or truncated
    env.close()

    damage = -tot_reward / _R_DAMAGE_TAKEN
    return {
        "dmg_ep":    damage / N_EPISODES,
        "wall_pct":  tot_wall / tot_decisions if tot_decisions else 0.0,
        "move_pct":  tot_moves / tot_decisions if tot_decisions else 0.0,
        "reward_ep": tot_reward / N_EPISODES,
        "shared":    shared,
    }


def main():
    os.makedirs("ai/logs", exist_ok=True)
    out_path = "ai/logs/eval_dodge.csv"
    rows = []
    print(f"=== GREEDY DODGE EVAL | vs AimShooter(0.7) Gunner bullets | {N_EPISODES} eps/role ===")
    header = f"{'algo':<8}{'role':<12}{'dmg/ep':>9}{'wall%':>8}{'move%':>8}{'reward/ep':>11}   (lower dmg/wall = better)"
    print(header)
    print("-" * (len(header) - 27))
    for algo in ALGOS:
        for role in ROLES:
            m = eval_one(algo, role)
            if m is None:
                print(f"{algo:<8}{role.__name__:<12}{'  (no weights)':>20}")
                continue
            name = role.__name__ + ("*" if m["shared"] else "")
            print(f"{algo:<8}{name:<12}{m['dmg_ep']:>9.1f}{m['wall_pct']*100:>7.1f}%"
                  f"{m['move_pct']*100:>7.1f}%{m['reward_ep']:>11.1f}")
            rows.append([algo, role.__name__, round(m["dmg_ep"], 2), round(m["wall_pct"], 4),
                         round(m["move_pct"], 4), round(m["reward_ep"], 2), int(m["shared"])])

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algo", "role", "dmg_taken_ep", "wall_pct", "move_pct", "reward_ep", "shared_gunner_policy"])
        w.writerows(rows)
    print("\n* = evaluated with the shared Gunner dodge policy (role has no own dodge weights)")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

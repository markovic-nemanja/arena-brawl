"""Comparison + algorithm-mechanics plots for the PPO vs DQN benchmark (aim / aim_move / dodge).

Figure groups written to ai/plots/:
    RESULTS   — final greedy-eval bars (the FAIR head-to-head: argmax, no exploration) 
                + behavioral bars (aim%, accuracy, fire%, wall%) that show WHETHER the skill was learned vs gamed.
                
    LEARNING  — episode reward vs environment steps
    
    MECHANICS — How each algorithm works, from its training logs (Gunner static aim as the representative run):
                * PPO: policy entropy (exploration -> commitment), policy loss (clipped surrogate), value loss (critic).
                * DQN: epsilon schedule (explore -> exploit), TD loss (Bellman error).
            Core difference: 
            DQN explores with an explicit epsilon schedule and learns a value function (Bellman).
            PPO explores through policy entropy and optimizes a clipped policy gradient.

NB: the eval BARS are the fair ranking (greedy). The training CURVES show learning speed/shape but DQN's reward
is epsilon-contaminated, so read curves for dynamics, bars for who-wins.

"""
import os
import csv
import collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGS = "ai/logs"
PLOTS = "ai/plots"
ALGOS = ["ppo", "dqn", "dueling", "a2c"]
COLOR = {"ppo": "#1f77b4", "dqn": "#ff7f0e", "dueling": "#2ca02c", "a2c": "#9467bd"}
ROLES = ["Gunner", "Bomber", "Dasher", "ToxicTrail", "Blackhole"]
MAX_ENTROPY = float(np.log(10)) # uniform policy over 10 discrete actions


# readers
def read_eval(path):
    rows = {}
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                rows[(r["algo"], r["role"])] = r
    return rows


def _log_path(algo, role, stage):
    """Resolve a training-log path, handling the legacy ppo_aim.csv (Gunner static) name."""
    p = os.path.join(LOGS, f"{algo}_{role.lower()}_{stage}.csv")
    if os.path.exists(p):
        return p
    legacy = os.path.join(LOGS, f"{algo}_{stage}.csv") # ppo_aim.csv == PPO Gunner static aim
    return legacy if os.path.exists(legacy) else None


def _read_log(path):
    """Return {column: np.array} for a training log (non-numeric cells -> nan)."""
    if not path or not os.path.exists(path):
        return None
    cols = collections.defaultdict(list)
    with open(path) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        for row in reader:
            for c in fields:
                try:
                    cols[c].append(float(row[c]))
                except (ValueError, TypeError):
                    cols[c].append(np.nan)
    return {c: np.array(v) for c, v in cols.items()}


def _smooth(y, k=11):
    return np.convolve(y, np.ones(k) / k, mode="valid") if len(y) >= k else y


def _reward_series(d, reward_cols):
    """Pull (steps, reward), dropping PPO rows with no completed episode (they log reward 0, an artifact)."""
    rcol = next((c for c in reward_cols if c in d), None)
    if rcol is None:
        return None, None
    reward, steps = d[rcol], d["steps"]
    keep = ~np.isnan(reward)
    if "episodes" in d:
        keep &= d["episodes"] > 0
    return steps[keep], reward[keep]


# results bars
def grouped_bar(evals, metric, title, ylabel, fname, lower_better=False):
    x = np.arange(len(ROLES))
    n = len(ALGOS)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, algo in enumerate(ALGOS):
        vals = []
        for role in ROLES:
            v = evals.get((algo, role), {}).get(metric)
            vals.append(float(v) if v not in (None, "") else np.nan)
        ax.bar(x + (i - (n - 1) / 2) * w, vals, w, label=algo.upper(), color=COLOR[algo])
    ax.set_xticks(x)
    ax.set_xticklabels(ROLES)
    ax.set_ylabel(ylabel)
    ax.set_title(title + ("   (lower = better)" if lower_better else ""))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, fname), dpi=120)
    plt.close(fig)
    print(f"  wrote {fname}")


# learning curves
def learning_curve_flagship(stage, reward_cols, title, fname):
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for algo in ALGOS:
        d = _read_log(_log_path(algo, "Gunner", stage))
        if d is None:
            continue
        steps, reward = _reward_series(d, reward_cols)
        if steps is None or len(steps) < 3:
            continue
        sm = _smooth(reward)
        ax.plot(steps[len(steps) - len(sm):], sm, label=algo.upper(), color=COLOR[algo], linewidth=1.8)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episode reward (smoothed)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, fname), dpi=120)
    plt.close(fig)
    print(f"  wrote {fname}")


def learning_curve_grid(stage, reward_cols, title, fname):
    fig, axes = plt.subplots(1, len(ROLES), figsize=(4 * len(ROLES), 4))
    plotted = False
    for ax, role in zip(axes, ROLES):
        for algo in ALGOS:
            d = _read_log(_log_path(algo, role, stage))
            if d is None:
                continue
            steps, reward = _reward_series(d, reward_cols)
            if steps is None or len(steps) < 3:
                continue
            sm = _smooth(reward)
            ax.plot(steps[len(steps) - len(sm):], sm, color=COLOR[algo], label=algo.upper(), linewidth=1.4)
            plotted = True
        ax.set_title(role)
        ax.set_xlabel("steps")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("episode reward (smoothed)")
    axes[0].legend()
    if not plotted:
        plt.close(fig)
        return
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, fname), dpi=120)
    plt.close(fig)
    print(f"  wrote {fname}")


# mechanics ("how it works")
def actor_critic_mechanics(algo, role, stage, title, fname):
    d = _read_log(_log_path(algo, role, stage))
    if d is None or "entropy" not in d:
        return
    steps = d["steps"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(steps, d["entropy"], color=COLOR[algo], linewidth=1.6)
    ax[0].axhline(MAX_ENTROPY, ls="--", color="gray", linewidth=0.8)
    ax[0].set_title("Policy entropy\n(exploration → commitment; dashed = max/uniform)")
    ax[1].plot(steps, d["policy_loss"], color="#2ca02c", linewidth=1.4)
    ax[1].set_title("Policy loss\n(policy gradient / clipped surrogate)")
    ax[2].plot(steps, d["value_loss"], color="#d62728", linewidth=1.4)
    ax[2].set_title("Value loss\n(critic, MSE)")
    for a in ax:
        a.set_xlabel("steps")
        a.grid(alpha=0.3)
    fig.suptitle(f"{algo.upper()} training dynamics — {title}")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, fname), dpi=120)
    plt.close(fig)
    print(f"  wrote {fname}")


def value_mechanics(algo, role, stage, title, fname):
    d = _read_log(_log_path(algo, role, stage))
    if d is None or "epsilon" not in d:
        return
    steps = d["steps"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(steps, d["epsilon"], color=COLOR[algo], linewidth=1.6)
    ax[0].set_ylim(0, 1.05)
    ax[0].set_title("Epsilon schedule\n(explicit explore → exploit)")
    ax[1].plot(steps, d["loss"], color="#d62728", linewidth=1.2)
    ax[1].set_title("TD loss\n(Bellman error)")
    for a in ax:
        a.set_xlabel("steps")
        a.grid(alpha=0.3)
    fig.suptitle(f"{algo.upper()} training dynamics — {title}")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, fname), dpi=120)
    plt.close(fig)
    print(f"  wrote {fname}")


# main
def main():
    os.makedirs(PLOTS, exist_ok=True)
    aim = read_eval(os.path.join(LOGS, "eval_aim.csv"))
    move = read_eval(os.path.join(LOGS, "eval_aim_move.csv"))
    dodge = read_eval(os.path.join(LOGS, "eval_dodge.csv"))

    print("results — final-performance bars:")
    grouped_bar(aim,   "kills_ep",     "Static aim — kills / episode",           "kills/ep",   "bar_aim_kills.png")
    grouped_bar(aim,   "accuracy",     "Static aim — accuracy (kills / shot)",   "kills/shot", "bar_aim_accuracy.png")
    grouped_bar(aim,   "aim_pct",      "Static aim — aim% (reads the target?)",  "aim% (0-1)", "bar_aim_aimpct.png")
    grouped_bar(aim,   "fire_pct",     "Static aim — fire fraction (spam check)", "fire% (0-1)", "bar_aim_firepct.png")
    grouped_bar(move,  "kills_ep",     "Moving aim — kills / episode",           "kills/ep",   "bar_move_kills.png")
    grouped_bar(dodge, "dmg_taken_ep", "Dodge — damage taken / episode",         "dmg/ep",     "bar_dodge_dmg.png", lower_better=True)
    grouped_bar(dodge, "wall_pct",     "Dodge — wall-hug fraction",              "wall% (0-1)", "bar_dodge_wall.png", lower_better=True)

    print("learning curves — flagship (Gunner) + per-role grid:")
    aim_cols = ["avg_hit_reward", "hit_reward", "avg_return_100", "avg_reward"]
    dodge_cols = ["avg_dodge_reward", "dodge_reward", "avg_return_100", "avg_reward"]
    learning_curve_flagship("aim",      aim_cols,   "Static aim learning — Gunner", "curve_aim_gunner.png")
    learning_curve_flagship("aim_move", aim_cols,   "Moving aim learning — Gunner", "curve_move_gunner.png")
    learning_curve_flagship("dodge",    dodge_cols, "Dodge learning — Gunner",      "curve_dodge_gunner.png")
    learning_curve_grid("aim",      aim_cols,   "Static aim learning — all roles", "grid_aim.png")
    learning_curve_grid("aim_move", aim_cols,   "Moving aim learning — all roles", "grid_move.png")
    learning_curve_grid("dodge",    dodge_cols, "Dodge learning — all roles",      "grid_dodge.png")

    print("mechanics — how each algorithm works (Gunner static aim):")
    actor_critic_mechanics("ppo", "Gunner", "aim", "Gunner static aim", "mech_ppo_gunner_aim.png")
    actor_critic_mechanics("a2c", "Gunner", "aim", "Gunner static aim", "mech_a2c_gunner_aim.png")
    value_mechanics("dqn", "Gunner", "aim", "Gunner static aim", "mech_dqn_gunner_aim.png")
    value_mechanics("dueling", "Gunner", "aim", "Gunner static aim", "mech_dueling_gunner_aim.png")

    print(f"\nall plots in {PLOTS}/")


if __name__ == "__main__":
    main()

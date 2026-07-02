import os
import csv
import time
import random
import wandb
from arena_env import ArenaBrawlEnv
from systems.roles import Gunner, Bomber, Dasher, ToxicTrail, Blackhole
from ai.rollout_buffer import RolloutBuffer
from ai.ppo_agent import PPOAgent
from systems.controller import EasyBot, MediumBot, HardBot

BOTS = [EasyBot, MediumBot]
BOT_PROB = 0.1

ROLES = [Gunner, Bomber, Dasher, ToxicTrail, Blackhole]
BASE = "ai/weights/ppo_{}_final.pth"

def freeze(learner):
    """
    
    """
    snap = PPOAgent()
    snap.network.load_state_dict(learner.network.state_dict())
    snap.network.eval()
    return snap

def _train_role(role, agent, pool, env, buffer, total_steps, rollout_size, rnd, log_dir):
    env.agent_role = role()
    
    log_path = os.path.join(log_dir, f"league_{role.__name__.lower()}.csv")
    first = not os.path.exists(log_path)
    log_file = open(log_path, "a", newline="")
    writer = csv.writer(log_file)
    if first:
        writer.writerow(["round", "update", "steps", "avg_reward", "win_rate", "entropy", "episodes"])

    wandb.init(project="arena-brawl-league-v2", name=f"{role.__name__}_r{rnd}", reinit=True)
    
    def new_opponent():
        if random.random() < BOT_PROB:
            env.opponent_agent = None
            env.opponent_bot = random.choice(BOTS)
            env.opponent_role = random.choice(ROLES)()
        else:
            opp_role, opp_brain = random.choice(pool)
            env.opponent_role = opp_role()
            env.opponent_agent = opp_brain

    new_opponent()
    state, _ = env.reset()

    episode_reward = 0.0
    completed = []
    wins = 0
    steps_done = 0
    update_num = 0
    start = last_print = time.time()

    while steps_done < total_steps:
        for _ in range(rollout_size):
            a, lp, v = agent.select_action(state)
            ns, r, term, trunc, _ = env.step(a)
            done = term or trunc
            buffer.store(state, a, r, done, lp, v)
            episode_reward += r
            state = ns
            steps_done += 1
            if done:
                completed.append(episode_reward)
                if term and env.opponent.hp <= 0:
                    wins += 1
                episode_reward = 0.0
                new_opponent()
                state, _ = env.reset()
            if steps_done >= total_steps:
                break

        last_value = agent.select_action(state)[2]
        pl, vl, ent = agent.update(buffer, last_value)
        buffer.clear()
        update_num += 1

        avg = sum(completed) / len(completed) if completed else 0
        wr = wins / len(completed) if completed else 0
        writer.writerow([rnd, update_num, steps_done, round(avg, 3), round(wr, 3), round(ent, 4), len(completed)])
        log_file.flush()
        wandb.log({"round": rnd, "steps": steps_done, "avg_reward": avg,
                   "win_rate": wr, "entropy": ent, "episodes": len(completed)})

        if time.time() - last_print >= 30:
            el = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
            print(f"[{el}] {role.__name__} r{rnd} | steps {steps_done} | win_rate {wr:.2f} "
                  f"| avg_reward {avg:.1f} | entropy {ent:.3f} | pool {len(pool)}")
            last_print = time.time()

        completed = []
        wins = 0

    log_file.close()
    wandb.finish()
    
def train_league(num_rounds=3, steps_per_round=400_000, rollout_size=2048,
                 save_dir="ai/weights", log_dir="ai/logs"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # one live learner per role, each loaded from its Stage-1 base brain
    learners = {role: PPOAgent(lr=1e-4) for role in ROLES}
    for role in ROLES:
        learners[role].load(BASE.format(role.__name__.lower()))

    # shared pool of (role_class, frozen_brain); seed it with all 5 base brains
    pool = [(role, freeze(learners[role])) for role in ROLES]

    env = ArenaBrawlEnv(agent_role=ROLES[0](), opponent_role=ROLES[0]())
    buffer = RolloutBuffer()

    for rnd in range(num_rounds):
        print(f"\n========== ROUND {rnd}  (pool size {len(pool)}) ==========")
        for role in ROLES:
            print(f"--- training {role.__name__} (round {rnd}) ---")
            _train_role(role, learners[role], pool, env, buffer,
                        steps_per_round, rollout_size, rnd, log_dir)
            
        # freeze every brain of each role and add to the pool for next round
        for role in ROLES:
            pool.append((role, freeze(learners[role])))
            learners[role].save(os.path.join(save_dir, f"ppo_{role.__name__.lower()}_league_r{rnd}.pth"))

    for role in ROLES:
        learners[role].save(os.path.join(save_dir, f"ppo_{role.__name__.lower()}_league_final.pth"))
    env.close()
    print("\nLeague complete -> ppo_<role>_league_final.pth saved for all roles.")


if __name__ == "__main__":
    train_league(num_rounds=3, steps_per_round=200_000)
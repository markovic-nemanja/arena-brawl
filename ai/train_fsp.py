import os
import csv
import random
from arena_env import ArenaBrawlEnv
from systems.roles import *
from systems.controller import EasyBot, MediumBot
from ai.replay_buffer import ReplayBuffer
from ai.dqn_agent import DQNAgent
import wandb
from collections import deque

ALL_ROLES = [Gunner, Bomber, Dasher, Blackhole, ToxicTrail, Splitter]

PHASE_BOUNDS = (500, 2000, 3000)

PHASE_CONFIG = {
    1: {"bot":EasyBot, "pool_prob":0.0},
    2: {"bot":MediumBot, "pool_prob":0.7},
    3: {"bot":MediumBot, "pool_prob":0.9}, # change later to HardBot
    4: {"bot":MediumBot, "pool_prob":0.9}, # bot stays as anti-forgetting anchor
}

def get_phase(episode):
    b1, b2, b3 = PHASE_BOUNDS
    if episode < b1:
        return 1
    elif episode < b2:
        return 2
    elif episode < b3:
        return 3
    return 4

def sample_recent_weighted(pool):
    weights = list(range(1, len(pool) + 1))
    return random.choices(pool, weights=weights, k=1)[0]

def train_fsp(
    agent_role=None,
    num_episodes=5000,
    buffer_capacity=100000,
    batch_size=64,
    save_dir="ai/weights",
    log_path="ai/logs/fsp_training.csv",
    snapshot_every=500,
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    env = ArenaBrawlEnv(agent_role=agent_role())
    agent = DQNAgent(batch_size=batch_size)
    buffer = ReplayBuffer(capacity=buffer_capacity)
    pool = []
    
    # live wandb logs
    recent_wins = deque(maxlen=100)
    wandb.init(
        project="arena-brawl-fsp",
        name=agent_role.__name__,
        config={
            "num_episodes": num_episodes,
            "batch_size": batch_size,
            "buffer_capacity": buffer_capacity,
            "gamma": agent.gamma,
            "epsilon_decay": agent.epsilon_decay,
            "phase_bounds": PHASE_BOUNDS,
        },
        reinit=True
    )
    
    log_file = open(log_path, mode='w', newline='')
    writer = csv.writer(log_file)
    writer.writerow(["episode", "phase", "opponent", "reward", "steps", "win", "epsilon", "loss"])
    
    for episode in range(num_episodes):
        phase = get_phase(episode)
        config = PHASE_CONFIG[phase]
        
        if len(pool) > 0 and random.random() < config["pool_prob"]:
            env.opponent_agent = sample_recent_weighted(pool)
            env.opponent_role = agent_role()
            opponent_label = "snapshot"
        else:
            env.opponent_agent = None
            env.opponent_bot = config["bot"]
            env.opponent_role = random.choice(ALL_ROLES)()
            opponent_label = config["bot"].__name__
            
        state, _ = env.reset()
        total_reward = 0
        total_loss = 0
        loss_count = 0
        step_count = 0
        win = 0
        
        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.store(state, action, reward, next_state, float(done))
            
            loss = agent.train_step(buffer)
            if loss is not None:
                total_loss += loss
                loss_count += 1
            
            total_reward += reward
            state = next_state
            step_count += 1
            
            if done:
                win = 1 if env.opponent.hp <= 0 else 0
                break
            
        agent.decay_epsilon()
            
        avg_loss = total_loss / loss_count if loss_count > 0 else 0
        writer.writerow([episode, phase, opponent_label, round(total_reward, 3), step_count, win, round(agent.epsilon, 4), round(avg_loss, 5)])

        recent_wins.append(win)
        wandb.log({
            "episode": episode,
            "phase": phase,
            "reward": total_reward,
            "win": win,
            "win_rate": sum(recent_wins) / len(recent_wins),
            "steps": step_count,
            "epsilon": agent.epsilon,
            "loss": avg_loss,
        })

        if (episode + 1) % snapshot_every == 0:
            path = os.path.join(save_dir, f"{agent_role.__name__.lower()}_snap_{episode + 1}.pth")
            agent.save(path)
            frozen = DQNAgent()
            frozen.load(path)
            frozen.epsilon = 0.0
            pool.append(frozen)
            print(f"Episode {episode + 1} | P{phase} | reward={total_reward:.1f} | win={win} | epsilon={agent.epsilon:.3f} | pool={len(pool)}")
                
    agent.save(os.path.join(save_dir, f"{agent_role.__name__.lower()}_final.pth"))
    log_file.close()
    env.close()
    wandb.finish() # end wandb
    print("FSP training completed.")
                
if __name__ == "__main__":
    for role in ALL_ROLES:
        print(f"=====Starting FSP training for role: {role.__name__}=====")
        train_fsp(agent_role=role, num_episodes=5000, log_path=f"ai/logs/fsp_{role.__name__.lower()}.csv",)
        print(f"=====Completed FSP training for role: {role.__name__}=====")
import os
import csv
import time

import wandb
from arena_env import ArenaBrawlEnv
from systems.roles import *
from systems.controller import StationaryBot, RandomBot, RandomShooterBot, EasyBot, MediumBot, HardBot
from ai.rollout_buffer import RolloutBuffer
from ai.ppo_agent import PPOAgent

PHASE_BOUNDS = (200000, 450000, 800000, 1400000, 2200000)
PHASE_BOTS = {
    1: StationaryBot,
    2: RandomBot,
    3: RandomShooterBot,
    4: EasyBot,
    5: MediumBot,
    6: HardBot,
}

def get_phase(steps):
    phase = 1
    for b in PHASE_BOUNDS:
        if steps >= b:
            phase += 1
        else:
            break
        
    return phase

def train_ppo(agent_role=Gunner, opponent_role=Gunner,
              total_steps=3000000, rollout_size=2048, save_dir="ai/weights", log_path="ai/logs/ppo_training.csv"):
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    env = ArenaBrawlEnv(agent_role=agent_role(), opponent_role=opponent_role())
    agent = PPOAgent()
    buffer = RolloutBuffer()
    
    wandb.init(
        project="arena-brawl-ppo",
        name=agent_role.__name__,
        config={
            "total_steps": total_steps,
            "rollout_size": rollout_size,
            "gamma": agent.gamma,
            "gae_lambda": agent.gae_lambda,
            "clip_epsilon": agent.clip_epsilon,
            "epochs": agent.epochs,
            "entropy_coef": agent.entropy_coef,
            "phase_bounds": PHASE_BOUNDS,
        },
        reinit=True
    )
    
    log_file = open(log_path, mode='w', newline="")
    writer = csv.writer(log_file)
    writer.writerow(["update", "steps", "phase", "avg_reward", "win_rate",
                     "entropy", "policy_loss", "value_loss", "episodes"])
    
    last_phase = get_phase(0)
    env.opponent_bot = PHASE_BOTS[last_phase]
    state, _ = env.reset()
    
    episode_reward = 0
    completed_rewards = []
    wins = 0
    steps_done = 0
    update_num = 0
    start_time = last_print = time.time()
    
    print(f"=== PPO {agent_role.__name__} | Phase {last_phase} "
          f"({PHASE_BOTS[last_phase].__name__}) ===")
    
    while steps_done < total_steps:
        # Collect rollout
        for _ in range(rollout_size):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            done = terminated or truncated
            buffer.store(state, action, reward, done, log_prob, value)
            
            episode_reward += reward
            state = next_state
            steps_done += 1
            
            if done:
                completed_rewards.append(episode_reward)
                if terminated and env.opponent.hp <= 0:
                    wins += 1
                episode_reward = 0
                phase = get_phase(steps_done)
                
                if phase != last_phase:
                    env.opponent_bot = PHASE_BOTS[phase]   # takes effect on this reset
                    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                    print(f"[{elapsed}] === {agent_role.__name__} | Phase {phase} "
                          f"({PHASE_BOTS[phase].__name__}) @ step {steps_done} ===")
                    last_phase = phase

                state, _ = env.reset()
            
            if steps_done >= total_steps:
                break
                    
        # Update
        last_value = agent.select_action(state)[2]
        policy_loss, value_loss, entropy = agent.update(buffer, last_value)
        buffer.clear()
        update_num += 1
        
        # Logging
        avg_reward = sum(completed_rewards) / len(completed_rewards) if completed_rewards else 0
        win_rate = wins / len(completed_rewards) if completed_rewards else 0
        phase = get_phase(steps_done)
        
        writer.writerow([update_num, steps_done, phase, round(avg_reward, 3),
                         round(win_rate, 3), round(entropy, 4),
                         round(policy_loss, 4), round(value_loss, 4),
                         len(completed_rewards)])
        log_file.flush()
        
        wandb.log({
            "update": update_num,
            "steps": steps_done,
            "phase": phase,
            "avg_reward": avg_reward,
            "win_rate": win_rate,
            "entropy": entropy,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "episodes": len(completed_rewards),
        })
        
        if time.time() - last_print >= 30:
            elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            print(f"[{elapsed}] {agent_role.__name__} | update {update_num} "
                  f"| Phase{phase} ({PHASE_BOTS[phase].__name__}) | steps {steps_done} "
                  f"| avg_reward {avg_reward:.2f} | win_rate {win_rate:.2f} "
                  f"| entropy {entropy:.3f} | episodes {len(completed_rewards)}")
            last_print = time.time()
            
        completed_rewards = []
        wins = 0
        
    agent.save(os.path.join(save_dir, f"ppo_{agent_role.__name__.lower()}_final.pth"))
    log_file.close()
    env.close()
    wandb.finish()
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"[{elapsed}] PPO {agent_role.__name__} DONE ({steps_done} steps).")
        
if __name__ == "__main__":
    ROLES = [Gunner, Bomber, Dasher, ToxicTrail, Blackhole]
    for role in ROLES:
        print(f"\n######## PPO training: {role.__name__} ########")
        
        try:
            train_ppo(
                agent_role=role,
                opponent_role=role,
                total_steps=3000000,
                log_path=f"ai/logs/ppo_{role.__name__.lower()}.csv",
            )
        except Exception as e:
            print(f"!!! {role.__name__} FAILED: {e} !!!")
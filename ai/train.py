import os
import csv
from arena_env import ArenaBrawlEnv
from systems.roles import *
from ai.replay_buffer import ReplayBuffer
from ai.dqn_agent import DQNAgent

def train(
    agent_role=None,
    opponent_role=None,
    num_episodes=5000,
    buffer_capacity=100000,
    batch_size=64,
    save_dir="ai/weights",
    log_path="ai/logs/dqn_training.csv",
    snapshot_every=500,
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    env = ArenaBrawlEnv(agent_role=agent_role, opponent_role=opponent_role)
    agent = DQNAgent(batch_size=batch_size)
    buffer = ReplayBuffer(capacity=buffer_capacity)
    
    log_file = open(log_path, mode='w', newline='')
    writer = csv.writer(log_file)
    writer.writerow(["episode", "reward", "steps", "win", "epsilon", "loss"])
    
    for episode in range(num_episodes):
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
        writer.writerow([episode, round(total_reward, 3), step_count, win, round(agent.epsilon, 4), round(avg_loss, 5)])
        
        if (episode + 1) % snapshot_every == 0:
            snapshot_path = os.path.join(save_dir, f"dqn_snapshot_{episode + 1}.pth")
            agent.save(snapshot_path)
            print(f"Episode {episode + 1} | reward={total_reward:.3f} |win={win} | epsilon={agent.epsilon:.4f} | loss={avg_loss:.5f}")
            
    agent.save(os.path.join(save_dir, "dqn_final.pth"))
    log_file.close()
    env.close()
    print("Training completed. Final model and logs saved.")
    
if __name__ == "__main__":
    train(
        agent_role=Gunner(),
        opponent_role=Bomber(),
        num_episodes=5000,
    )
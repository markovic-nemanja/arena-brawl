import os, csv, time
from arena_env import ArenaBrawlEnv
from systems.roles import *
from systems.controller import StationaryBot
from ai.replay_buffer import ReplayBuffer
from ai.dueling_dqn_agent import DuelingDQNAgent


def train_aim_dueling(agent_role=Gunner, total_steps=1_000_000, opponent_bot=StationaryBot,
                      base_weights=None, buffer_capacity=100000, batch_size=64,
                      save_dir="ai/weights", log_path="ai/logs/dueling_aim.csv", save_suffix="aim",
                      checkpoint_every=100_000):
    os.makedirs(save_dir, exist_ok=True)
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    env = ArenaBrawlEnv(agent_role=agent_role(), opponent_role=agent_role(),
                        opponent_bot=opponent_bot, aim_practice=True)
    # Aim episodes always last 1,350 decisions, so the general 0.999 episode
    # decay would still leave ~48% random actions after one million steps.
    agent = DuelingDQNAgent(batch_size=batch_size, epsilon_decay=0.995)
    if base_weights:
        agent.load(base_weights)
        print(f"[transfer] loaded {base_weights}")
    buffer = ReplayBuffer(capacity=buffer_capacity)

    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["episode", "steps", "hit_reward", "epsilon", "loss"])

    steps_done = 0
    episode = 0
    last_checkpoint = 0
    start_time = last_print = time.time()
    print(f"=== AIM-DUELING {agent_role.__name__} vs {opponent_bot.__name__} (target practice) ===")

    while steps_done < total_steps:
        state, _ = env.reset()
        ep_reward = 0.0
        total_loss = 0.0
        loss_count = 0
        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.store(state, action, reward, next_state, float(done))
            loss = agent.train_step(buffer)
            if loss is not None:
                total_loss += loss
                loss_count += 1
            ep_reward += reward
            state = next_state
            steps_done += 1

            checkpoint = steps_done // checkpoint_every if checkpoint_every > 0 else 0
            if checkpoint > last_checkpoint:
                last_checkpoint = checkpoint
                checkpoint_path = os.path.join(
                    save_dir,
                    f"dueling_{agent_role.__name__.lower()}_{save_suffix}_{steps_done}.pth",
                )
                agent.save(checkpoint_path)

            if done or steps_done >= total_steps:
                break

        agent.decay_epsilon()
        episode += 1
        avg_loss = total_loss / loss_count if loss_count > 0 else 0
        writer.writerow([episode, steps_done, round(ep_reward, 3), round(agent.epsilon, 4), round(avg_loss, 5)])
        log_file.flush()

        if time.time() - last_print >= 30:
            elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            print(f"[{elapsed}] {agent_role.__name__} | ep {episode} | steps {steps_done} "
                  f"| hit_reward {ep_reward:.0f} | eps {agent.epsilon:.3f} | loss {avg_loss:.4f}")
            last_print = time.time()

    agent.save(os.path.join(save_dir, f"dueling_{agent_role.__name__.lower()}_{save_suffix}.pth"))
    log_file.close()
    env.close()
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"[{elapsed}] AIM-DUELING {agent_role.__name__} DONE ({steps_done} steps).")


if __name__ == "__main__":
    for role in [Gunner, Bomber, Dasher, ToxicTrail, Blackhole]:
        print(f"\n######## Dueling aim: {role.__name__} ########")
        train_aim_dueling(agent_role=role, total_steps=1_000_000,
                          log_path=f"ai/logs/dueling_{role.__name__.lower()}_aim.csv")

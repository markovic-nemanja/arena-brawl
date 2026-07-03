import argparse
import csv
import os
import random
import time
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
import wandb

from ai.ppo_agent import PPOAgent
from ai.rollout_buffer import RolloutBuffer
from arena_env import ArenaBrawlEnv
from systems.controller import (
    EasyBot,
    GentleAggressor,
    MediumBot,
    RandomBot,
    RandomShooterBot,
    StationaryBot,
)
from systems.roles import Blackhole, Bomber, Dasher, Gunner, ToxicTrail


ALL_ROLES = [Gunner, Bomber, Dasher, ToxicTrail, Blackhole]


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    opponent_mix: tuple
    evaluation_bot: object
    required_win_rate: float
    randomize_opponent_role: bool


# Every stage retains part of the previous distribution to reduce catastrophic
# forgetting. Advancement is based on greedy evaluation, never elapsed steps.
CURRICULUM = (
    CurriculumStage(
        "stationary",
        ((StationaryBot, 1.0),),
        StationaryBot,
        0.80,
        False,
    ),
    CurriculumStage(
        "moving_target",
        ((StationaryBot, 0.25), (RandomBot, 0.75)),
        RandomBot,
        0.80,
        False,
    ),
    CurriculumStage(
        "random_shooter",
        ((RandomBot, 0.25), (RandomShooterBot, 0.75)),
        RandomShooterBot,
        0.70,
        False,
    ),
    CurriculumStage(
        "gentle_40",
        (
            (RandomShooterBot, 0.25),
            (partial(GentleAggressor, fire_prob=0.4), 0.75),
        ),
        partial(GentleAggressor, fire_prob=0.4),
        0.70,
        True,
    ),
    CurriculumStage(
        "gentle_70",
        (
            (partial(GentleAggressor, fire_prob=0.4), 0.25),
            (partial(GentleAggressor, fire_prob=0.7), 0.75),
        ),
        partial(GentleAggressor, fire_prob=0.7),
        0.65,
        True,
    ),
    CurriculumStage(
        "easy",
        (
            (partial(GentleAggressor, fire_prob=0.7), 0.25),
            (EasyBot, 0.75),
        ),
        EasyBot,
        0.60,
        True,
    ),
    CurriculumStage(
        "medium",
        ((EasyBot, 0.30), (MediumBot, 0.70)),
        MediumBot,
        0.55,
        True,
    ),
)


def choose_weighted_factory(weighted_factories):
    factories = [factory for factory, _ in weighted_factories]
    weights = [weight for _, weight in weighted_factories]
    return random.choices(factories, weights=weights, k=1)[0]


def configure_opponent(env, stage, learner_role):
    env.opponent_agent = None
    env.opponent_bot = choose_weighted_factory(stage.opponent_mix)
    opponent_role = (
        random.choice(ALL_ROLES)
        if stage.randomize_opponent_role
        else learner_role
    )
    env.opponent_role = opponent_role()


def evaluate_policy(
    agent,
    learner_role,
    stage,
    episodes=50,
    deterministic=True,
    seed=10_000,
):
    """Evaluate without allowing evaluation randomness to change training."""
    python_random_state = random.getstate()
    numpy_random_state = np.random.get_state()
    torch_random_state = torch.random.get_rng_state()
    cuda_random_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )

    env = ArenaBrawlEnv(
        agent_role=learner_role(),
        opponent_role=learner_role(),
        opponent_bot=stage.evaluation_bot,
    )
    wins = losses = draws = 0
    hp_differences = []
    action_counts = np.zeros(env.action_space.n, dtype=np.int64)

    try:
        for episode in range(episodes):
            episode_seed = seed + episode
            random.seed(episode_seed)
            np.random.seed(episode_seed)

            env.opponent_role = (
                random.choice(ALL_ROLES)()
                if stage.randomize_opponent_role
                else learner_role()
            )
            state, _ = env.reset(seed=episode_seed)
            done = False

            while not done:
                action = agent.act(state, deterministic=deterministic)
                action_counts[action] += 1
                state, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

            if terminated and env.opponent.hp <= 0:
                wins += 1
            elif terminated and env.agent.hp <= 0:
                losses += 1
            else:
                draws += 1
            hp_differences.append(env.agent.hp - env.opponent.hp)
    finally:
        env.close()
        random.setstate(python_random_state)
        np.random.set_state(numpy_random_state)
        torch.random.set_rng_state(torch_random_state)
        if cuda_random_states is not None:
            torch.cuda.set_rng_state_all(cuda_random_states)

    total_actions = max(1, int(action_counts.sum()))
    return {
        "win_rate": wins / episodes,
        "loss_rate": losses / episodes,
        "draw_rate": draws / episodes,
        "mean_hp_difference": float(np.mean(hp_differences)),
        "ability_fraction": float(action_counts[9]) / total_actions,
        "stationary_fraction": float(
            action_counts[0] + action_counts[9]
        )
        / total_actions,
        "action_counts": action_counts.tolist(),
    }


def train_ppo(
    agent_role=Gunner,
    rollout_size=2048,
    max_steps_per_stage=600_000,
    evaluation_every=100_000,
    evaluation_episodes=50,
    save_dir="ai/weights",
    log_path=None,
    resume_path=None,
):
    os.makedirs(save_dir, exist_ok=True)
    if log_path is None:
        log_path = f"ai/logs/ppo_{agent_role.__name__.lower()}.csv"
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    env = ArenaBrawlEnv(agent_role=agent_role())
    agent = PPOAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
    )
    if resume_path is not None:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(
                f"Resume checkpoint does not exist: {resume_path}"
            )
        agent.load(resume_path)
        print(f"Resumed {agent_role.__name__} from {resume_path}")
    buffer = RolloutBuffer()

    wandb.init(
        project="arena-brawl-ppo-v2",
        name=agent_role.__name__,
        config={
            "rollout_size": rollout_size,
            "max_steps_per_stage": max_steps_per_stage,
            "evaluation_every": evaluation_every,
            "evaluation_episodes": evaluation_episodes,
            "observation_size": env.observation_space.shape[0],
            "action_size": env.action_space.n,
            "entropy_coef": agent.entropy_coef,
            "resume_path": resume_path,
        },
        reinit=True,
    )

    log_file = open(log_path, mode="w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(
        [
            "stage",
            "update",
            "total_steps",
            "stage_steps",
            "training_reward",
            "training_win_rate",
            "training_loss_rate",
            "training_draw_rate",
            "entropy",
            "policy_loss",
            "value_loss",
            "greedy_win_rate",
            "greedy_loss_rate",
            "greedy_draw_rate",
            "greedy_hp_difference",
            "ability_fraction",
            "stationary_fraction",
        ]
    )

    total_steps = 0
    update_number = 0
    start_time = time.time()

    try:
        for stage_index, stage in enumerate(CURRICULUM, start=1):
            print(
                f"\n=== {agent_role.__name__}: stage "
                f"{stage_index}/{len(CURRICULUM)} ({stage.name}) ==="
            )
            stage_steps = 0
            last_evaluation = 0
            consecutive_mastery_evaluations = 0
            stage_mastered = False

            if resume_path is not None and stage_index == 1:
                resume_result = evaluate_policy(
                    agent,
                    agent_role,
                    stage,
                    episodes=evaluation_episodes,
                    deterministic=True,
                    seed=10_000 + stage_index * 1_000,
                )
                print(
                    "resume_check "
                    f"greedy_win={resume_result['win_rate']:.1%} "
                    f"draw={resume_result['draw_rate']:.1%}"
                )
                if resume_result["win_rate"] >= stage.required_win_rate:
                    consecutive_mastery_evaluations = 1

            configure_opponent(env, stage, agent_role)
            state, _ = env.reset()
            episode_reward = 0.0
            completed_rewards = []
            training_wins = training_losses = training_draws = 0

            while stage_steps < max_steps_per_stage:
                for _ in range(rollout_size):
                    action, log_probability, value = agent.select_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(
                        action
                    )
                    done = terminated or truncated
                    buffer.store(
                        state,
                        action,
                        reward,
                        done,
                        log_probability,
                        value,
                    )
                    episode_reward += reward
                    state = next_state
                    total_steps += 1
                    stage_steps += 1

                    if done:
                        completed_rewards.append(episode_reward)
                        if terminated and env.opponent.hp <= 0:
                            training_wins += 1
                        elif terminated and env.agent.hp <= 0:
                            training_losses += 1
                        else:
                            training_draws += 1

                        episode_reward = 0.0
                        configure_opponent(env, stage, agent_role)
                        state, _ = env.reset()

                    if stage_steps >= max_steps_per_stage:
                        break

                last_value = (
                    0.0 if buffer.dones[-1] else agent.get_value(state)
                )
                policy_loss, value_loss, entropy = agent.update(
                    buffer, last_value
                )
                buffer.clear()
                update_number += 1

                completed_episodes = (
                    training_wins + training_losses + training_draws
                )
                average_reward = (
                    float(np.mean(completed_rewards))
                    if completed_rewards
                    else 0.0
                )
                training_win_rate = (
                    training_wins / completed_episodes
                    if completed_episodes
                    else 0.0
                )
                training_loss_rate = (
                    training_losses / completed_episodes
                    if completed_episodes
                    else 0.0
                )
                training_draw_rate = (
                    training_draws / completed_episodes
                    if completed_episodes
                    else 0.0
                )
                greedy_result = None

                # rollout_size does not necessarily divide evaluation_every.
                # Always evaluate at the stage limit so a final qualifying
                # result is not skipped just short of the next interval.
                should_evaluate = (
                    stage_steps - last_evaluation >= evaluation_every
                    or stage_steps >= max_steps_per_stage
                )
                if should_evaluate:
                    last_evaluation = stage_steps
                    greedy_result = evaluate_policy(
                        agent,
                        agent_role,
                        stage,
                        episodes=evaluation_episodes,
                        deterministic=True,
                        seed=10_000 + stage_index * 1_000,
                    )
                    sampled_result = evaluate_policy(
                        agent,
                        agent_role,
                        stage,
                        episodes=max(20, evaluation_episodes // 2),
                        deterministic=False,
                        seed=20_000 + stage_index * 1_000,
                    )

                    print(
                        f"steps={total_steps} "
                        f"greedy_win={greedy_result['win_rate']:.1%} "
                        f"sampled_win={sampled_result['win_rate']:.1%} "
                        f"draw={greedy_result['draw_rate']:.1%} "
                        f"ability={greedy_result['ability_fraction']:.1%} "
                        f"stationary={greedy_result['stationary_fraction']:.1%}"
                    )
                    agent.save(
                        os.path.join(
                            save_dir,
                            f"ppo_{agent_role.__name__.lower()}_"
                            f"{stage.name}_{total_steps}.pth",
                        )
                    )

                    if greedy_result["win_rate"] >= stage.required_win_rate:
                        consecutive_mastery_evaluations += 1
                    else:
                        consecutive_mastery_evaluations = 0

                    if consecutive_mastery_evaluations >= 2:
                        stage_mastered = True
                        agent.save(
                            os.path.join(
                                save_dir,
                                f"ppo_{agent_role.__name__.lower()}_"
                                f"{stage.name}_mastered.pth",
                            )
                        )

                writer.writerow(
                    [
                        stage.name,
                        update_number,
                        total_steps,
                        stage_steps,
                        round(average_reward, 4),
                        round(training_win_rate, 4),
                        round(training_loss_rate, 4),
                        round(training_draw_rate, 4),
                        round(entropy, 4),
                        round(policy_loss, 4),
                        round(value_loss, 4),
                        round(greedy_result["win_rate"], 4)
                        if greedy_result
                        else "",
                        round(greedy_result["loss_rate"], 4)
                        if greedy_result
                        else "",
                        round(greedy_result["draw_rate"], 4)
                        if greedy_result
                        else "",
                        round(greedy_result["mean_hp_difference"], 3)
                        if greedy_result
                        else "",
                        round(greedy_result["ability_fraction"], 4)
                        if greedy_result
                        else "",
                        round(greedy_result["stationary_fraction"], 4)
                        if greedy_result
                        else "",
                    ]
                )
                log_file.flush()

                wandb.log(
                    {
                        "stage": stage_index,
                        "total_steps": total_steps,
                        "stage_steps": stage_steps,
                        "training_reward": average_reward,
                        "training_win_rate": training_win_rate,
                        "training_loss_rate": training_loss_rate,
                        "training_draw_rate": training_draw_rate,
                        "entropy": entropy,
                        "policy_loss": policy_loss,
                        "value_loss": value_loss,
                        **(
                            {
                                "greedy_win_rate": greedy_result["win_rate"],
                                "greedy_draw_rate": greedy_result["draw_rate"],
                                "ability_fraction": greedy_result[
                                    "ability_fraction"
                                ],
                                "stationary_fraction": greedy_result[
                                    "stationary_fraction"
                                ],
                            }
                            if greedy_result
                            else {}
                        ),
                    }
                )

                completed_rewards.clear()
                training_wins = training_losses = training_draws = 0

                if stage_mastered:
                    elapsed = time.strftime(
                        "%H:%M:%S", time.gmtime(time.time() - start_time)
                    )
                    print(
                        f"[{elapsed}] Mastered {stage.name} after "
                        f"{stage_steps} steps."
                    )
                    break

            if not stage_mastered:
                agent.save(
                    os.path.join(
                        save_dir,
                        f"ppo_{agent_role.__name__.lower()}_"
                        f"{stage.name}_not_mastered.pth",
                    )
                )
                print(
                    f"Stopped: {agent_role.__name__} did not master "
                    f"{stage.name}; it will not be advanced automatically."
                )
                return False

        agent.save(
            os.path.join(
                save_dir, f"ppo_{agent_role.__name__.lower()}_final.pth"
            )
        )
        return True
    finally:
        log_file.close()
        env.close()
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        help=(
            "Load policy weights for the first role from this checkpoint "
            "before continuing training."
        ),
    )
    args = parser.parse_args()

    for role_index, role in enumerate((Gunner, ToxicTrail)):
        successful = train_ppo(
            agent_role=role,
            resume_path=args.resume if role_index == 0 else None,
        )
        if not successful:
            break

import argparse
import csv
import os
import random
import subprocess
import time
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import torch
import wandb

from ai.ppo_agent import PPOAgent
from ai.rollout_buffer import VectorRolloutBuffer
from arena_env import ArenaBrawlEnv, create_vector_env
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
ROLE_BY_NAME = {role.__name__.lower(): role for role in ALL_ROLES}

# The primary experiment is mirror-role combat. Opponent behavior becomes more
# capable while the body/ability remains the same as the learning agent.
PHASES = {
    1: {
        "name": "stationary",
        "mix": ((StationaryBot, 1.0),),
        "gate": 0.75,
    },
    2: {
        "name": "random_movement",
        "mix": ((RandomBot, 1.0),),
        "gate": 0.65,
    },
    3: {
        "name": "random_shooter",
        "mix": ((RandomShooterBot, 1.0),),
        "gate": 0.55,
    },
    4: {
        "name": "gentle_50",
        "mix": ((partial(GentleAggressor, fire_prob=0.50), 1.0),),
        "gate": 0.45,
    },
    5: {
        "name": "easy",
        "mix": ((EasyBot, 1.0),),
        "gate": 0.35,
    },
    6: {
        "name": "easy_medium",
        "mix": ((EasyBot, 0.40), (MediumBot, 0.60)),
        "gate": 0.30,
    },
}


def phase_training_mix(phase, rehearsal_fraction=0.30):
    """Use mostly the current tier while retaining previous opponents."""
    current = PHASES[phase]["mix"]
    if phase == 1:
        return current

    result = [
        (factory, weight * (1.0 - rehearsal_fraction))
        for factory, weight in current
    ]
    previous_entries = [
        (factory, weight)
        for previous_phase in range(1, phase)
        for factory, weight in PHASES[previous_phase]["mix"]
    ]
    previous_total = sum(weight for _, weight in previous_entries)
    result.extend(
        (factory, rehearsal_fraction * weight / previous_total)
        for factory, weight in previous_entries
    )
    return tuple(result)


@dataclass
class Curriculum:
    phase: int = 1
    phase_start_step: int = 0
    pass_streak: int = 0
    required_passes: int = 2
    minimum_phase_steps: int = 100_000
    maximum_phase_steps: int = 300_000
    best_win_rates: dict = field(default_factory=dict)
    last_promotion_reason: str = ""

    def record_evaluation(self, win_rate, global_step):
        self.best_win_rates[self.phase] = max(
            win_rate, self.best_win_rates.get(self.phase, 0.0)
        )
        if win_rate >= PHASES[self.phase]["gate"]:
            self.pass_streak += 1
        else:
            self.pass_streak = 0

        phase_steps = global_step - self.phase_start_step
        mastered = (
            self.phase < max(PHASES)
            and phase_steps >= self.minimum_phase_steps
            and self.pass_streak >= self.required_passes
        )
        reached_cap = (
            self.phase < max(PHASES)
            and phase_steps >= self.maximum_phase_steps
        )
        if mastered or reached_cap:
            self.last_promotion_reason = "mastery" if mastered else "step_cap"
            self.phase += 1
            self.phase_start_step = global_step
            self.pass_streak = 0
            return True
        return False

    def state_dict(self):
        return {
            "phase": self.phase,
            "phase_start_step": self.phase_start_step,
            "pass_streak": self.pass_streak,
            "required_passes": self.required_passes,
            "minimum_phase_steps": self.minimum_phase_steps,
            "maximum_phase_steps": self.maximum_phase_steps,
            "best_win_rates": self.best_win_rates,
            "last_promotion_reason": self.last_promotion_reason,
        }

    @classmethod
    def from_state_dict(cls, state):
        return cls(**state) if state else cls()


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _completed_wins(terminations, dones, infos):
    final_info = infos.get("final_info", {})
    opponent_hp = final_info.get("opponent_hp")
    if opponent_hp is None:
        return 0
    return sum(
        1
        for index in np.flatnonzero(dones)
        if terminations[index] and opponent_hp[index] <= 0
    )


def evaluate_phase(agent, role, phase, episodes=100, seed=1_000_000):
    """Greedy, fixed-seed validation against the current phase only."""
    env = ArenaBrawlEnv(
        agent_role=role(),
        opponent_role=role(),
        opponent_mix=PHASES[phase]["mix"],
        opponent_roles=(role,),
    )
    wins = timeouts = 0
    damage_dealt = damage_taken = 0.0
    wall_blocks = casts = 0
    aim_total = 0.0
    decisions = 0
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + phase * 10_000 + episode)
            done = False
            while not done:
                action = agent.act(state)
                state, _, terminated, truncated, info = env.step(action)
                damage_dealt += info["damage_dealt"]
                damage_taken += info["damage_taken"]
                wall_blocks += int(info["wall_blocked"])
                casts += int(info["ability_cast"])
                aim_total += float(info["aim_alignment"])
                decisions += 1
                done = terminated or truncated
            wins += int(terminated and env.opponent.hp <= 0)
            timeouts += int(truncated)
    finally:
        env.close()

    return {
        "win_rate": wins / episodes,
        "timeout_rate": timeouts / episodes,
        "damage_dealt": damage_dealt / episodes,
        "damage_taken": damage_taken / episodes,
        "wall_block_rate": wall_blocks / max(decisions, 1),
        "casts_per_episode": casts / episodes,
        "cast_aim": aim_total / max(casts, 1),
    }


def train_ppo(
    agent_role=Gunner,
    total_steps=2_000_000,
    rollout_steps=256,
    num_envs=20,
    minibatch_size=256,
    asynchronous=True,
    save_dir="ai/weights",
    log_path="ai/logs/ppo_training.csv",
    seed=0,
    eval_interval=50_000,
    eval_episodes=100,
    checkpoint_interval=250_000,
    wandb_mode="online",
    device=None,
    resume_path=None,
    reset_phase_progress=False,
):
    if num_envs != 20:
        raise ValueError(
            "The controlled comparison is configured for exactly 20 vector environments."
        )
    if total_steps % num_envs:
        raise ValueError("total_steps must be divisible by num_envs")
    rollout_size = rollout_steps * num_envs
    if minibatch_size > rollout_size:
        raise ValueError("minibatch_size cannot exceed the rollout batch")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    agent = PPOAgent(batch_size=minibatch_size, device=device)
    buffer = VectorRolloutBuffer()
    curriculum = Curriculum()
    steps_done = 0
    update_number = 0
    if resume_path:
        checkpoint = agent.load(resume_path, load_optimizer=True)
        steps_done = int(checkpoint.get("global_step", 0))
        update_number = int(checkpoint.get("update", 0))
        curriculum = Curriculum.from_state_dict(checkpoint.get("curriculum"))
        # Old checkpoints predate the corrected gates/cap. Always use the
        # current curriculum policy while retaining their learned phase.
        curriculum.required_passes = 2
        curriculum.minimum_phase_steps = 100_000
        curriculum.maximum_phase_steps = 300_000
        if reset_phase_progress:
            curriculum.phase_start_step = steps_done
            curriculum.pass_streak = 0

    config = {
        "role": agent_role.__name__,
        "total_steps": total_steps,
        "rollout_steps": rollout_steps,
        "rollout_size": rollout_size,
        "num_envs": num_envs,
        "minibatch_size": minibatch_size,
        "eval_interval": eval_interval,
        "eval_episodes": eval_episodes,
        "seed": seed,
        "observation_size": 31,
        "action_size": 18,
        "git_commit": _git_commit(),
    }
    wandb.init(
        project="arena-brawl-ppo-v2",
        name=f"{agent_role.__name__}-seed-{seed}",
        config=config,
        mode=wandb_mode,
        reinit=True,
    )

    append_log = bool(resume_path and os.path.exists(log_path))
    log_file = open(log_path, mode="a" if append_log else "w", newline="")
    writer = csv.writer(log_file)
    if not append_log:
        writer.writerow(
            [
                "update", "steps", "phase", "phase_name", "avg_reward",
                "win_rate", "episodes", "policy_loss", "value_loss",
                "entropy", "approx_kl", "clip_fraction",
                "explained_variance", "learning_rate", "wall_block_rate",
                "casts_per_episode", "cast_aim", "num_envs",
            ]
        )

    def save_checkpoint(filename):
        agent.save(
            os.path.join(save_dir, filename),
            global_step=steps_done,
            update=update_number,
            role=agent_role.__name__,
            curriculum=curriculum.state_dict(),
            config=config,
            observation_version=2,
            action_version=2,
            torch_rng_state=torch.get_rng_state(),
        )

    vector_env = None
    states = None
    active_phase = None
    episode_rewards = np.zeros(num_envs, dtype=np.float64)
    next_eval_step = ((steps_done // eval_interval) + 1) * eval_interval
    next_checkpoint = (
        (steps_done // checkpoint_interval) + 1
    ) * checkpoint_interval
    start_time = last_print = time.time()

    try:
        print(
            f"PPO {agent_role.__name__}: {num_envs} environments, "
            f"rollout={rollout_size}, device={agent.device}"
        )
        while steps_done < total_steps:
            if curriculum.phase != active_phase:
                if vector_env is not None:
                    vector_env.close()
                active_phase = curriculum.phase
                vector_env, states = create_vector_env(
                    agent_role,
                    phase_training_mix(active_phase),
                    (agent_role,),
                    num_envs,
                    seed + active_phase * 10_000,
                    asynchronous=asynchronous,
                )
                episode_rewards.fill(0.0)
                print(
                    f"PPO {agent_role.__name__}: phase {active_phase} "
                    f"({PHASES[active_phase]['name']})"
                )

            vector_steps = min(
                rollout_steps,
                (total_steps - steps_done) // num_envs,
            )
            buffer.clear()
            completed_rewards = []
            wins = 0
            casts = wall_blocks = decisions = 0
            aim_total = 0.0

            for _ in range(vector_steps):
                actions, log_probabilities, values = agent.select_actions(states)
                next_states, rewards, terminations, truncations, infos = (
                    vector_env.step(actions)
                )
                dones = np.logical_or(terminations, truncations)
                buffer.store(
                    states,
                    actions,
                    rewards,
                    dones,
                    log_probabilities,
                    values,
                )
                episode_rewards += rewards
                wins += _completed_wins(terminations, dones, infos)
                for index in np.flatnonzero(dones):
                    completed_rewards.append(float(episode_rewards[index]))
                    episode_rewards[index] = 0.0

                casts_now = np.asarray(
                    infos.get("ability_cast", np.zeros(num_envs)), dtype=np.float32
                )
                casts += int(casts_now.sum())
                wall_blocks += int(np.asarray(
                    infos.get("wall_blocked", np.zeros(num_envs)), dtype=np.float32
                ).sum())
                aim_total += float(np.asarray(
                    infos.get("aim_alignment", np.zeros(num_envs)), dtype=np.float32
                ).sum())
                decisions += num_envs
                states = next_states
                steps_done += num_envs

            learning_rate = agent.set_training_progress(steps_done / total_steps)
            last_values = agent.get_values(states)
            update_metrics = agent.update(buffer, last_values)
            update_number += 1

            average_reward = (
                float(np.mean(completed_rewards)) if completed_rewards else 0.0
            )
            win_rate = wins / len(completed_rewards) if completed_rewards else 0.0
            training_metrics = {
                "update": update_number,
                "steps": steps_done,
                "phase": curriculum.phase,
                "avg_reward": average_reward,
                "win_rate": win_rate,
                "episodes": len(completed_rewards),
                "learning_rate": learning_rate,
                "wall_block_rate": wall_blocks / max(decisions, 1),
                "casts_per_episode": casts / max(len(completed_rewards), 1),
                "cast_aim": aim_total / max(casts, 1),
                **update_metrics,
            }
            writer.writerow(
                [
                    update_number,
                    steps_done,
                    curriculum.phase,
                    PHASES[curriculum.phase]["name"],
                    round(average_reward, 5),
                    round(win_rate, 5),
                    len(completed_rewards),
                    round(update_metrics["policy_loss"], 6),
                    round(update_metrics["value_loss"], 6),
                    round(update_metrics["entropy"], 6),
                    round(update_metrics["approx_kl"], 6),
                    round(update_metrics["clip_fraction"], 6),
                    round(update_metrics["explained_variance"], 6),
                    learning_rate,
                    training_metrics["wall_block_rate"],
                    training_metrics["casts_per_episode"],
                    training_metrics["cast_aim"],
                    num_envs,
                ]
            )
            log_file.flush()
            wandb.log({f"train/{key}": value for key, value in training_metrics.items()})

            if steps_done >= next_checkpoint:
                save_checkpoint(
                    f"ppo_{agent_role.__name__.lower()}_v2_step_{steps_done}.pth"
                )
                save_checkpoint(f"ppo_{agent_role.__name__.lower()}_v2_latest.pth")
                next_checkpoint += checkpoint_interval

            if steps_done >= next_eval_step:
                evaluated_phase = curriculum.phase
                evaluation = evaluate_phase(
                    agent,
                    agent_role,
                    evaluated_phase,
                    episodes=eval_episodes,
                    seed=seed + 2_000_000,
                )
                previous_best = curriculum.best_win_rates.get(evaluated_phase, -1.0)
                if evaluation["win_rate"] > previous_best:
                    save_checkpoint(
                        f"ppo_{agent_role.__name__.lower()}_v2_phase"
                        f"{evaluated_phase}_best.pth"
                    )
                promoted = curriculum.record_evaluation(
                    evaluation["win_rate"], steps_done
                )
                wandb.log({
                    "steps": steps_done,
                    "eval/phase": evaluated_phase,
                    "eval/gate": PHASES[evaluated_phase]["gate"],
                    "eval/pass_streak": curriculum.pass_streak,
                    "eval/promotion_reason": curriculum.last_promotion_reason,
                    **{f"eval/{key}": value for key, value in evaluation.items()},
                })
                print(
                    f"EVAL phase {evaluated_phase} | win "
                    f"{evaluation['win_rate']:.1%} | timeout "
                    f"{evaluation['timeout_rate']:.1%} | gate "
                    f"{PHASES[evaluated_phase]['gate']:.0%}"
                    + (
                        f" | PROMOTED ({curriculum.last_promotion_reason})"
                        if promoted else ""
                    )
                )
                next_eval_step += eval_interval

            if time.time() - last_print >= 30:
                elapsed = time.strftime(
                    "%H:%M:%S", time.gmtime(time.time() - start_time)
                )
                print(
                    f"[{elapsed}] PPO {agent_role.__name__} | phase "
                    f"{curriculum.phase} | steps {steps_done} | reward "
                    f"{average_reward:.3f} | win {win_rate:.1%} | entropy "
                    f"{update_metrics['entropy']:.3f} | SPS "
                    f"{steps_done / max(time.time() - start_time, 1e-6):.0f}"
                )
                last_print = time.time()

        save_checkpoint(f"ppo_{agent_role.__name__.lower()}_v2_final.pth")
        save_checkpoint(f"ppo_{agent_role.__name__.lower()}_v2_latest.pth")
    finally:
        if vector_env is not None:
            vector_env.close()
        log_file.close()
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train mastery-gated PPO in 20 parallel mirror matches."
    )
    parser.add_argument("--role", choices=ROLE_BY_NAME, default="gunner")
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=20)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=250_000)
    parser.add_argument("--save-dir", default="ai/weights")
    parser.add_argument("--log-path", default=None)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--reset-phase-progress", action="store_true")
    parser.add_argument("--sync-envs", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    role = ROLE_BY_NAME[args.role]
    train_ppo(
        agent_role=role,
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        num_envs=args.num_envs,
        minibatch_size=args.minibatch_size,
        asynchronous=not args.sync_envs,
        seed=args.seed,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        checkpoint_interval=args.checkpoint_interval,
        save_dir=args.save_dir,
        wandb_mode=args.wandb_mode,
        device=args.device,
        resume_path=args.resume,
        reset_phase_progress=args.reset_phase_progress,
        log_path=args.log_path or f"ai/logs/ppo_{args.role}_v2.csv",
    )

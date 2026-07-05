"""Watch a DODGE-trained agent weave through incoming fire (rendered) — any algorithm.

Set ROLE and ALGORITHM below. The opponent is a moving AimShooter(0.7) firing Gunner bullets.
Matching train_dodge_*.py — so you can see whether the policy actually sidesteps hazards and keeps off the walls (walls deal damage too).

Agents are greedy (no exploration).

"""
import os
import pygame
from arena_env import ArenaBrawlEnv
from systems.roles import Gunner, Bomber, Dasher, ToxicTrail, Blackhole
from systems.controller import AimShooterBot
from ai.ppo_agent import PPOAgent
from ai.dqn_agent import DQNAgent
from ai.dueling_dqn_agent import DuelingDQNAgent

ROLE = Gunner # Gunner / Bomber / Dasher / ToxicTrail / Blackhole
ALGORITHM = "ppo" # "ppo" / "dqn" / "dueling"

# each algorithm -> how to build a fresh agent (Dueling still defaults to state_size=20, so force 22)
AGENTS = {
    "ppo": lambda: PPOAgent(),
    "dqn": lambda: DQNAgent(),
    "dueling": lambda: DuelingDQNAgent(state_size=22),
}

weights = f"ai/weights/{ALGORITHM}_{ROLE.__name__.lower()}_dodge.pth"
if not os.path.exists(weights):
    raise SystemExit(f"[watch_dodge] no weights at {weights} — "
                     f"train {ALGORITHM} dodge for {ROLE.__name__} first, or pick another ROLE/ALGORITHM.")

agent = AGENTS[ALGORITHM]()
agent.load(weights)
print(f"[watch_dodge] {ALGORITHM.upper()} {ROLE.__name__} — greedy, dodging a moving AimShooter(0.7)")

env = ArenaBrawlEnv(agent_role=ROLE(), opponent_role=Gunner(),
                    opponent_bot=lambda: AimShooterBot(0.7), dodge_practice=True, render_mode="human")
state, _ = env.reset()
running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_q):
            running = False
    state, reward, terminated, truncated, _ = env.step(agent.act(state))   # greedy
    if terminated or truncated:
        state, _ = env.reset()
env.close()

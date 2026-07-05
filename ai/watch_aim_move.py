"""Watch a STATIONARY-aim agent play versus respawning targets (rendered) — any algorithm.

Set ROLE and ALGORITHM below. 
The target is a StationaryBot that respawns at a new random spot on each hit
This is result from train_aim_move_*.py files
"""

import os
import pygame
from arena_env import ArenaBrawlEnv
from systems.roles import Gunner, Bomber, Dasher, ToxicTrail, Blackhole
from systems.controller import RandomBot
from ai.ppo_agent import PPOAgent
from ai.dqn_agent import DQNAgent
from ai.dueling_dqn_agent import DuelingDQNAgent

ROLE = Blackhole # Gunner / Bomber / Dasher / ToxicTrail / Blackhole
ALGORITHM = "ppo" # "ppo" / "dqn" / "dueling"

# each algorithm -> how to build a fresh agent
AGENTS = {
    "ppo": lambda: PPOAgent(),
    "dqn": lambda: DQNAgent(),
    "dueling": lambda: DuelingDQNAgent(state_size=22),
}

weights = f"ai/weights/{ALGORITHM}_{ROLE.__name__.lower()}_aim_move.pth"
if not os.path.exists(weights):
    raise SystemExit(f"[watch_aim_move] no weights at {weights} — "
                     f"train {ALGORITHM} moving aim for {ROLE.__name__} first, or pick another ROLE/ALGORITHM.")

agent = AGENTS[ALGORITHM]()
agent.load(weights)
print(f"[watch_aim_move] {ALGORITHM.upper()} {ROLE.__name__} — greedy, MOVING target (RandomBot), respawns on each hit")

env = ArenaBrawlEnv(agent_role=ROLE(), opponent_role=ROLE(), opponent_bot=RandomBot,
                    aim_practice=True, render_mode="human")
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

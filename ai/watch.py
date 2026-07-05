"""Watch a STATIONARY-aim agent snap onto respawning targets (rendered) — any algorithm.

Set ROLE and ALGO below. The target is a StationaryBot that respawns at a new random spot on each
hit (aim_practice), matching train_aim_ppo.py / train_aim_dqn.py / train_aim_dueling.py. All three
agents expose the same greedy API (`agent.act(state)` -> action int, `agent.load(path)`), so the only
per-algorithm difference is which class to build and which weights file to load.

Run:  python -m ai.watch_aim   (Q or close the window to quit)"""
import os
import pygame
from arena_env import ArenaBrawlEnv
from systems.roles import Gunner, Bomber, Dasher, ToxicTrail, Blackhole
from systems.controller import StationaryBot
from ai.ppo_agent import PPOAgent
from ai.dqn_agent import DQNAgent
from ai.dueling_dqn_agent import DuelingDQNAgent
from ai.a2c_agent import A2CAgent

ROLE = ToxicTrail   # Gunner / Bomber / Dasher / ToxicTrail / Blackhole
ALGORITHM = "dueling"       # "ppo" / "dqn" / "dueling"

AGENTS = {
    "ppo": lambda: PPOAgent(),
    "dqn": lambda: DQNAgent(),
    "dueling": lambda: DuelingDQNAgent(),
    "a2c": lambda: A2CAgent(),
}

weights = f"ai/weights/{ALGORITHM}_{ROLE.__name__.lower()}_aim.pth"
if not os.path.exists(weights):
    raise SystemExit(f"[watch_aim] no weights at {weights} — "
                     f"train {ALGORITHM} stationary aim for {ROLE.__name__} first, or pick another ROLE/ALGORITHM.")

agent = AGENTS[ALGORITHM]()
agent.load(weights)
print(f"[watch_aim] {ALGORITHM.upper()} {ROLE.__name__} — greedy, target respawns on each hit")

env = ArenaBrawlEnv(agent_role=ROLE(), opponent_role=ROLE(), opponent_bot=StationaryBot,
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
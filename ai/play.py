import pygame
from arena_env import ArenaBrawlEnv
from systems.roles import *
from ai.ppo_agent import PPOAgent
from ai.dqn_agent import DQNAgent
from ai.dueling_dqn_agent import DuelingDQNAgent

# --- choose what to watch ---
ROLE = ToxicTrail # role to play against
ALGORITHM = "ppo" # "ppo" or "dqn"
STAGE = "v2_final" # weight-file suffix (e.g. "v2_final", "league")
WEIGHTS = f"ai/weights/{ALGORITHM}_{ROLE.__name__.lower()}_{STAGE}.pth"

# Greedy opponent: the env drives the opponent by calling .select_action().
# We return the agent's argmax (act()) = its actual decision, not a random sample.
class Greedy:
    def __init__(self, agent):
        self.agent = agent
    def select_action(self, state):
        return self.agent.act(state)

AGENT_CLASSES = {
    "dqn": DQNAgent,
    "dueling_dqn": DuelingDQNAgent,
    "ppo": PPOAgent,
}

agent = AGENT_CLASSES[ALGORITHM]()
agent.load(WEIGHTS)
print(f"[play]  {ALGORITHM.upper()}  body={ROLE.__name__}  brain={WEIGHTS}")   # body and brain must match

env = ArenaBrawlEnv(agent_role=ROLE(), opponent_role=ROLE(),
                    opponent_agent=Greedy(agent), render_mode="human")

# keyboard -> action index 0-9 (same scheme as RLController)
_DIR = {(0, 0): 0, (0, -1): 1, (0, 1): 2, (-1, 0): 3, (1, 0): 4,
        (-1, -1): 5, (1, -1): 6, (-1, 1): 7, (1, 1): 8}

def keys_to_action(keys):
    dx = (1 if keys[pygame.K_d] or keys[pygame.K_RIGHT] else 0) - (1 if keys[pygame.K_a] or keys[pygame.K_LEFT] else 0)
    dy = (1 if keys[pygame.K_s] or keys[pygame.K_DOWN] else 0) - (1 if keys[pygame.K_w] or keys[pygame.K_UP] else 0)
    movement_action = _DIR[(dx, dy)]
    return movement_action + (9 if keys[pygame.K_SPACE] else 0)

state, _ = env.reset()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    action = keys_to_action(pygame.key.get_pressed())
    state, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        state, _ = env.reset()
env.close()

import pygame
from arena_env import ArenaBrawlEnv
from ai.dqn_agent import DQNAgent
from systems.roles import Gunner, Bomber

# --- load the trained model (use your latest snapshot) ---
model = DQNAgent()
model.load("ai/weights/gunner_final_fsp.pth")
model.epsilon = 0.0                      # greedy — play its best

env = ArenaBrawlEnv(
    agent_role=Gunner(),                 # YOU (purple, left)
    opponent_role=Gunner(),              # the model (orange, right) — mirror
    opponent_agent=model,
    render_mode="human",
)

# map keyboard → action index 0-9 (same scheme as RLController)
_DIR = {(0,0):0,(0,-1):1,(0,1):2,(-1,0):3,(1,0):4,(-1,-1):5,(1,-1):6,(-1,1):7,(1,1):8}
def keys_to_action(keys):
    if keys[pygame.K_SPACE]:
        return 9                         # ability (fires in last move direction)
    dx = (1 if keys[pygame.K_d] or keys[pygame.K_RIGHT] else 0) - (1 if keys[pygame.K_a] or keys[pygame.K_LEFT] else 0)
    dy = (1 if keys[pygame.K_s] or keys[pygame.K_DOWN]  else 0) - (1 if keys[pygame.K_w] or keys[pygame.K_UP]   else 0)
    return _DIR[(dx, dy)]

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
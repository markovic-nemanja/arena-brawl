import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import settings as S
from entities.player import Player
from systems.controller import EasyBot, RLController
from systems.roles import *
from game import Game
import random

class ArenaBrawlEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    _ARENA_WIDTH = S.ARENA_RIGHT - S.ARENA_LEFT
    _ARENA_HEIGHT = S.ARENA_BOTTOM - S.ARENA_TOP
    _MAX_DISTANCE = math.hypot(_ARENA_WIDTH, _ARENA_HEIGHT)
    
    def __init__(self, agent_role=None, opponent_role=None, render_mode=None, max_steps=5400,
                 opponent_agent=None, opponent_bot=EasyBot):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.frame_skip = 4
        self.current_step = 0
        self.agent_role = agent_role or Gunner()
        self.opponent_role = opponent_role or Gunner()
        self.opponent_agent = opponent_agent
        self.opponent_bot = opponent_bot
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(16,), dtype=np.float32)
        self.action_space = spaces.Discrete(10)
        
        self._rl_controller = RLController()
        self._opponent_controller = RLController()
        self._screen = None
        self._clock = None
        self.game = None
        self.agent = None
        self.opponent = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._rl_controller.current_action = 0
        self.current_step = 0
        
        px = random.randint(S.ARENA_LEFT + S.PLAYER_RADIUS, S.ARENA_RIGHT - S.PLAYER_RADIUS)
        py = random.randint(S.ARENA_TOP + S.PLAYER_RADIUS, S.ARENA_BOTTOM - S.PLAYER_RADIUS)
        
        while True:
            ox = random.randint(S.ARENA_LEFT + S.PLAYER_RADIUS, S.ARENA_RIGHT - S.PLAYER_RADIUS)
            oy = random.randint(S.ARENA_TOP + S.PLAYER_RADIUS, S.ARENA_BOTTOM - S.PLAYER_RADIUS)
            if math.hypot(ox - px, oy - py) > 200: # don't spawn on top of each other
                break
            
        self.agent = Player(
            x=px, y=py,
            color=S.PURPLE,
            controller=self._rl_controller,
            role=type(self.agent_role)()
        )
        
        if self.opponent_agent is not None:
            self._opponent_controller.current_action = 0
            opponent_controller = self._opponent_controller
        else:
            opponent_controller = self.opponent_bot()
            
        self.opponent = Player(
            x=ox, y=oy,
            color=S.ORANGE,
            controller=opponent_controller,
            role=type(self.opponent_role)()
        )
        
        self.game = Game(self.agent, self.opponent)
        
        if self.render_mode == "human" and self._screen is None:
            self._init_pygame()
        
        return self._get_obs(self.agent, self.opponent), {}
    
    def _potential(self):
        dx = self.opponent.x - self.agent.x
        dy = self.opponent.y - self.agent.y
        distance = math.hypot(dx, dy)
        
        closeness = 1 - (distance / self._MAX_DISTANCE)
        aim = (self.agent.last_direction.x * (dx /distance) + self.agent.last_direction.y * (dy / distance)) if distance > 0 else 0
        
        return 0.3 * closeness + 0.5 * aim
    
    def _step(self):
        self.current_step += 1
        
        if self.opponent_agent is not None:
            self._opponent_controller.current_action = self.opponent_agent.select_action(self._get_obs(self.opponent, self.agent))
            
        phi_before = self._potential()
        prev_hp = self.agent.hp
        prev_opponent_hp = self.opponent.hp
        
        winner = self.game.step(keys=None, dt=1/S.FPS)
        
        phi_after = self._potential()
        damage_dealt = prev_opponent_hp - self.opponent.hp
        damage_taken = prev_hp - self.agent.hp
        
        reward = damage_dealt * 0.3 - damage_taken * 0.1 - 0.001
        reward += 0.99 * phi_after - phi_before
        
        terminated = winner is not None
        truncated = self.current_step >= self.max_steps
        
        if terminated:
            reward += 30 if winner is self.agent else -30
        
        if self.render_mode == "human":
            self.render()
            
        return reward, terminated, truncated
    
    # frame skip - one agent decision = 4 game frames
    # summing up rewards over 4 frames into a single step -> 1/4 decisions per fight
    # adding for faster learning and to see if it fixes the problem of the agent not learning to attack
    def step(self, action):
        self._rl_controller.current_action = int(action)
        total_reward = 0.0
        terminated = truncated = False
        
        for _ in range(self.frame_skip):
            reward, terminated, truncated = self._step()
            total_reward += reward

            if terminated or truncated:
                break
            
        return self._get_obs(self.agent, self.opponent), total_reward, terminated, truncated, {}
    
    def render(self):
        self.game.render(self._screen)
        pygame.display.flip()
        self._clock.tick(S.FPS)
        
    def close(self):
        if self._screen is not None:
            pygame.quit()
            self._screen = None
            self._clock = None
            
    def _init_pygame(self):
        pygame.init()
        self._screen = pygame.display.set_mode((S.SCREEN_W, S.SCREEN_H))
        pygame.display.set_caption("Arena Brawl - RL")
        self._clock = pygame.time.Clock()
        
    def _get_obs(self, me, them):
        p, o = me, them
        
        px = (p.x - S.ARENA_LEFT) / self._ARENA_WIDTH
        py = (p.y - S.ARENA_TOP)  / self._ARENA_HEIGHT
        ox = (o.x - S.ARENA_LEFT) / self._ARENA_WIDTH
        oy = (o.y - S.ARENA_TOP)  / self._ARENA_HEIGHT
        
        php = np.clip(p.hp / S.PLAYER_MAX_HP, 0, 1)
        ohp = np.clip(o.hp / S.PLAYER_MAX_HP, 0, 1)
        
        pcd = min(p.cooldown / p.role.cooldown, 1) if p.role.cooldown > 0 else 0 # min-max normalization [a,b] -> [0,1] : (value-a)/(b-a)
        ocd = min(o.cooldown / o.role.cooldown, 1) if o.role.cooldown > 0 else 0 # min-max normalization
        
        distance = math.hypot(p.x - o.x, p.y - o.y) / self._MAX_DISTANCE
        
        max_velocity = S.PLAYER_SPEED * math.sqrt(2)
        ovx = np.clip(o.vx / max_velocity * 0.5 + 0.5, 0.0, 1.0) # min-max normalization
        ovy = np.clip(o.vy / max_velocity * 0.5 + 0.5, 0.0, 1.0) # min-max normalization
        
        h1x, h1y, h2x, h2y = self._get_hazards(p, o)
        
        immobile = 1 if o.stun_timer > 0 else 0
        
        return np.array([
            px, py, ox, oy,
            php, ohp,
            pcd, ocd,
            distance,
            ovx, ovy,
            h1x, h1y, h2x, h2y,
            immobile   
        ], dtype=np.float32)
        
    def _get_hazards(self, me, them):
        p = me
        hazards = []
        
        for projectile in them.projectiles:
            if not projectile.get("alive", False):
                continue
            
            hx = (projectile["x"] - S.ARENA_LEFT) / self._ARENA_WIDTH
            hy = (projectile["y"] - S.ARENA_TOP) / self._ARENA_HEIGHT
            d = math.hypot(p.x - projectile["x"], p.y - projectile["y"])
            hazards.append((d, hx, hy))
            
        hazards.sort()
        h1x = hazards[0][1] if len(hazards) > 0 else 0.5
        h1y = hazards[0][2] if len(hazards) > 0 else 0.5
        h2x = hazards[1][1] if len(hazards) > 1 else 0.5
        h2y = hazards[1][2] if len(hazards) > 1 else 0.5
        return h1x, h1y, h2x, h2y
        
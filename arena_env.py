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
        
        P1X, P2X, Y = 226, 494, 403
        self._rl_controller.current_action = 0
        self.current_step = 0
        
        self.agent = Player(
            x=P1X, y=Y,
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
            x=P2X, y=Y,
            color=S.ORANGE,
            controller=opponent_controller,
            role=type(self.opponent_role)()
        )
        
        self.game = Game(self.agent, self.opponent)
        
        if self.render_mode == "human" and self._screen is None:
            self._init_pygame()
        
        return self._get_obs(self.agent, self.opponent), {}
    
    def step(self, action):
        self._rl_controller.current_action = int(action)
        self.current_step += 1
        
        if self.opponent_agent is not None:
            opponent_obs = self._get_obs(self.opponent, self.agent)
            self._opponent_controller.current_action = self.opponent_agent.select_action(opponent_obs)
        
        prev_agent_hp = self.agent.hp
        prev_opponent_hp = self.opponent.hp
        
        winner = self.game.step(keys=None, dt=1/S.FPS)
        
        damage_dealt = prev_opponent_hp - self.opponent.hp
        damage_taken = prev_agent_hp - self.agent.hp
        
        reward = damage_dealt * 0.1 - damage_taken * 0.1 - 0.001
        terminated = winner is not None
        truncated = self.current_step >= self.max_steps
        
        if terminated:
            reward += 10.0 if winner is self.agent else -10.0
            
        if self.render_mode == "human":
            self.render()
            
        return self._get_obs(self.agent, self.opponent), reward, terminated, truncated, {}
    
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
        
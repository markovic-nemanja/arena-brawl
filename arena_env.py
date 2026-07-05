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
from collections import deque

# --- reward system ---

_R_DAMAGE_DEALT = 0.3
_R_DAMAGE_TAKEN = 0.1
_R_WIN = 10
_R_LOSE = 10
_R_TRUNCATED = 3

# NUDGES
_R_PROXIMITY = 0.5
_PROX_RADIUS = 70

_R_DODGE = 0.5
_DODGE_RADIUS = 120
_DODGE_AIM = 0.7

_R_STATIC = 0.005
_STATIC_WINDOW = 15
_STATIC_MIN_DISP = 70

_R_FIRE_MOVE = 0.2
_FIRE_MOVE_WINDOW = 4

_R_NO_FIRE = 0.005
_NO_FIRE_PATIENCE = 30

_AIM_DOT = 0.7

class ArenaBrawlEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    _ARENA_WIDTH = S.ARENA_RIGHT - S.ARENA_LEFT
    _ARENA_HEIGHT = S.ARENA_BOTTOM - S.ARENA_TOP
    _MAX_DISTANCE = math.hypot(_ARENA_WIDTH, _ARENA_HEIGHT)
    
    def __init__(self, agent_role=None, opponent_role=None, render_mode=None, max_steps=5400,
                 opponent_agent=None, opponent_bot=EasyBot, aim_practice=False):
        super().__init__()
        self.aim_practice = aim_practice
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.frame_skip = 4
        self.current_step = 0
        self.agent_role = agent_role or Gunner()
        self.opponent_role = opponent_role or Gunner()
        self.opponent_agent = opponent_agent
        self.opponent_bot = opponent_bot
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(22,), dtype=np.float32)
        self.action_space = spaces.Discrete(10)
        
        self._rl_controller = RLController()
        self._opponent_controller = RLController()
        self._screen = None
        self._clock = None
        self.game = None
        self.agent = None
        self.opponent = None
        
        self._MAX_HAZARD_SPEED = 500
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._rl_controller.current_action = 0
        self.current_step = 0
        
        px = self.np_random.integers(
            S.ARENA_LEFT + S.PLAYER_RADIUS,
            S.ARENA_RIGHT - S.PLAYER_RADIUS + 1,
        )
        py = self.np_random.integers(
            S.ARENA_TOP + S.PLAYER_RADIUS,
            S.ARENA_BOTTOM - S.PLAYER_RADIUS + 1,
        )
        
        while True:
            ox = self.np_random.integers(
                S.ARENA_LEFT + S.PLAYER_RADIUS,
                S.ARENA_RIGHT - S.PLAYER_RADIUS + 1,
            )
            oy = self.np_random.integers(
                S.ARENA_TOP + S.PLAYER_RADIUS,
                S.ARENA_BOTTOM - S.PLAYER_RADIUS + 1,
            )
            if math.hypot(ox - px, oy - py) > 200:
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

        # behaviour-reward tracking (fresh each episode)
        self._pos_history = deque(maxlen=_STATIC_WINDOW)
        self._fire_move_counter = 0
        self._since_fire = 0

        return self._get_obs(self.agent, self.opponent), {}
    
    def _step(self):
        self.current_step += 1

        prev_hp = self.agent.hp
        prev_opponent_hp = self.opponent.hp

        winner = self.game.step(keys=None, dt=1/S.FPS)

        damage_dealt = prev_opponent_hp - self.opponent.hp
        damage_taken = prev_hp - self.agent.hp

        if self.aim_practice:
            reward = _R_DAMAGE_DEALT * damage_dealt
            if self.opponent.hp <= 0:
                self._respawn_target()
            self.agent.hp = S.PLAYER_MAX_HP
            truncated = self.current_step >= self.max_steps
            if self.render_mode == "human":
                self.render()
            return reward, False, truncated, 0.0

        terminated = winner is not None
        truncated = self.current_step >= self.max_steps

        reward = _R_DAMAGE_DEALT * damage_dealt - _R_DAMAGE_TAKEN * damage_taken

        if terminated:
            reward += _R_WIN if winner is self.agent else -_R_LOSE
        elif truncated:
            reward -= _R_TRUNCATED

        if self.render_mode == "human":
            self.render()

        return reward, terminated, truncated, damage_taken

    def _respawn_target(self):
        """Teleport the (stationary) target to a fresh random spot and heal it — used in aim_practice."""
        while True:
            ox = self.np_random.integers(
                S.ARENA_LEFT + S.PLAYER_RADIUS,
                S.ARENA_RIGHT - S.PLAYER_RADIUS + 1,
            )
            oy = self.np_random.integers(
                S.ARENA_TOP + S.PLAYER_RADIUS,
                S.ARENA_BOTTOM - S.PLAYER_RADIUS + 1,
            )
            if math.hypot(ox - self.agent.x, oy - self.agent.y) > 200:
                break
        self.opponent.x = ox
        self.opponent.y = oy
        self.opponent.hp = S.PLAYER_MAX_HP
        self.opponent.projectiles = []
        self.opponent.cooldown = 0
        self.opponent.stun_timer = 0
        # Do not let hazards aimed at the old target damage a freshly spawned one.
        # This is especially important for trapping blackholes, whose damage phase
        # remains active after the killing blow.
        self.agent.projectiles = []

    def step(self, action):
        self._rl_controller.current_action = int(action)
        if self.opponent_agent is not None:
            opp_action = self.opponent_agent.select_action(self._get_obs(self.opponent, self.agent))
            if isinstance(opp_action, tuple):
                opp_action = opp_action[0]
            self._opponent_controller.current_action = int(opp_action)

        cd_before = self.agent.cooldown
        total_reward = 0.0
        damage_taken_dec = 0.0
        terminated = truncated = False

        for _ in range(self.frame_skip):
            reward, terminated, truncated, dmg_taken = self._step()
            total_reward += reward
            damage_taken_dec += dmg_taken
            if terminated or truncated:
                break

        if not terminated and not truncated and not self.aim_practice:
            total_reward += self._behaviour_rewards(int(action), cd_before, damage_taken_dec)

        return self._get_obs(self.agent, self.opponent), total_reward, terminated, truncated, {}

    def _behaviour_rewards(self, action, cd_before, damage_taken_dec):
        """Per-decision tactical nudges: reward landing hazards on the enemy and dodging,
        discourage standing still / going quiet, reward the shoot-then-reposition pattern."""
        r = 0.0
        moved = 1 <= action <= 8
        fired_shot = action == 9 and cd_before <= 0

        dx = self.opponent.x - self.agent.x
        dy = self.opponent.y - self.agent.y
        dist = math.hypot(dx, dy)
        aimed = dist > 0 and (self.agent.last_direction.x * (dx/dist) + self.agent.last_direction.y * (dy/dist)) >= _AIM_DOT

        for p in self.agent.projectiles:
            if p.get("alive") and "x" in p and not p.get("_prox_reward"):
                if math.hypot(p["x"] - self.opponent.x, p["y"] - self.opponent.y) < _PROX_RADIUS:
                    r += _R_PROXIMITY
                    p["_prox_reward"] = True

        for p in self.opponent.projectiles:
            if not p.get("alive") or "x" not in p:
                continue
            vx = p.get("dx", 0); vy = p.get("dy", 0); speed = math.hypot(vx, vy)
            tox = self.agent.x - p["x"]; toy = self.agent.y - p["y"]; d = math.hypot(tox, toy)
            if speed == 0 or d == 0:
                continue
            if d < _DODGE_RADIUS and (vx/speed * (tox/d) + vy/speed * (toy/d)) >= _DODGE_AIM:
                p["_threat"] = True
            elif p.get("_threat") and not p.get("_dodge_reward") and d >= _DODGE_RADIUS and damage_taken_dec <= 0:
                r += _R_DODGE
                p["_dodge_reward"] = True

        self._pos_history.append((self.agent.x, self.agent.y))
        if len(self._pos_history) >= _STATIC_WINDOW:
            ox, oy = self._pos_history[0]
            if math.hypot(self.agent.x - ox, self.agent.y - oy) < _STATIC_MIN_DISP:
                r -= _R_STATIC

        if fired_shot and aimed:
            self._fire_move_counter = _FIRE_MOVE_WINDOW
        elif self._fire_move_counter > 0:
            if moved:
                r += _R_FIRE_MOVE
                self._fire_move_counter = 0
            else:
                self._fire_move_counter -= 1

        if fired_shot:
            self._since_fire = 0
        else:
            self._since_fire += 1
            if self.agent.cooldown <= 0 and self._since_fire > _NO_FIRE_PATIENCE:
                r -= _R_NO_FIRE

        return r
    
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

        player_x = (p.x - S.ARENA_LEFT) / self._ARENA_WIDTH
        player_y = (p.y - S.ARENA_TOP)  / self._ARENA_HEIGHT
        player_hp = np.clip(p.hp / S.PLAYER_MAX_HP, 0, 1)
        player_cd = min(p.cooldown / p.role.cooldown, 1) if p.role.cooldown > 0 else 0
        player_facing_x = p.last_direction.x * 0.5 + 0.5
        player_facing_y = p.last_direction.y * 0.5 + 0.5

        opponent_relative_x = np.clip((o.x - p.x) / self._ARENA_WIDTH * 0.5 + 0.5, 0, 1)
        opponent_relative_y = np.clip((o.y - p.y) / self._ARENA_HEIGHT * 0.5 + 0.5, 0, 1)
        opponent_hp = np.clip(o.hp / S.PLAYER_MAX_HP, 0, 1)
        opponent_cd = min(o.cooldown / o.role.cooldown, 1) if o.role.cooldown > 0 else 0
        opponent_vx = o.last_direction.x * 0.5 + 0.5
        opponent_vy = o.last_direction.y * 0.5 + 0.5

        dxo = p.x - o.x
        dyo = p.y - o.y
        dist = math.hypot(dxo, dyo)
        rng = p.role.attack_range
        range_ratio = np.clip(dist / (2.0 * rng), 0.0, 1.0) if rng > 0 else 1.0

        h1x, h1y, h1vx, h1vy, h2x, h2y, h2vx, h2vy = self._get_hazards(p, o)
        
        opp_aiming = 0.0
        if dist > 0 and o.cooldown <= 0:
            facing_dot = o.last_direction.x * (dxo / dist) + o.last_direction.y * (dyo / dist)
            opp_aiming = 1.0 if facing_dot > 0.7 else 0.0

        return np.array([
            player_x, player_y, opponent_relative_x, opponent_relative_y,
            player_hp, opponent_hp,
            player_cd, opponent_cd,
            player_facing_x, player_facing_y,
            range_ratio,
            opponent_vx, opponent_vy,
            h1x, h1y, h1vx, h1vy, h2x, h2y, h2vx, h2vy,
            opp_aiming
        ], dtype=np.float32)

    def _get_hazards(self, me, them):
        p = me
        hazards = []

        for projectile in them.projectiles:
            if not projectile.get("alive", False):
                continue

            d = math.hypot(p.x - projectile["x"], p.y - projectile["y"])
            hx = np.clip((projectile["x"] - p.x) / self._ARENA_WIDTH  * 0.5 + 0.5, 0.0, 1.0)
            hy = np.clip((projectile["y"] - p.y) / self._ARENA_HEIGHT * 0.5 + 0.5, 0.0, 1.0)
            hvx = np.clip(projectile.get("dx", 0) / self._MAX_HAZARD_SPEED * 0.5 + 0.5, 0.0, 1.0)
            hvy = np.clip(projectile.get("dy", 0) / self._MAX_HAZARD_SPEED * 0.5 + 0.5, 0.0, 1.0)
            hazards.append((d, hx, hy, hvx, hvy))

        hazards.sort(key=lambda h: h[0])
        h1x  = hazards[0][1] if len(hazards) > 0 else 0.5
        h1y  = hazards[0][2] if len(hazards) > 0 else 0.5
        h1vx = hazards[0][3] if len(hazards) > 0 else 0.5
        h1vy = hazards[0][4] if len(hazards) > 0 else 0.5
        h2x  = hazards[1][1] if len(hazards) > 1 else 0.5
        h2y  = hazards[1][2] if len(hazards) > 1 else 0.5
        h2vx = hazards[1][3] if len(hazards) > 1 else 0.5
        h2vy = hazards[1][4] if len(hazards) > 1 else 0.5
        return h1x, h1y, h1vx, h1vy, h2x, h2y, h2vx, h2vy

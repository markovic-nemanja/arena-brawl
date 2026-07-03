import math

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

import settings as S
from entities.player import Player
from game import Game
from systems.controller import EasyBot, RLController
from systems.roles import Blackhole, Bomber, Dasher, Gunner, Splitter, ToxicTrail


class ArenaBrawlEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    _ARENA_WIDTH = S.ARENA_RIGHT - S.ARENA_LEFT
    _ARENA_HEIGHT = S.ARENA_BOTTOM - S.ARENA_TOP
    _MAX_DISTANCE = math.hypot(_ARENA_WIDTH, _ARENA_HEIGHT)
    _MAX_HAZARD_SPEED = 600.0

    ROLE_CLASSES = (Gunner, Bomber, Dasher, Blackhole, ToxicTrail, Splitter)
    ROLE_INDEX = {role_class: index for index, role_class in enumerate(ROLE_CLASSES)}

    # self position (2), opponent relative position (2), HP/cooldowns (4),
    # self facing/movement (4), opponent facing/movement (4), distance/stun (2),
    # opponent role (6), and two hazards x 5 values = 34.
    OBSERVATION_SIZE = 34

    def __init__(
        self,
        agent_role=None,
        opponent_role=None,
        render_mode=None,
        max_steps=5400,
        opponent_agent=None,
        opponent_bot=EasyBot,
    ):
        super().__init__()
        self.render_mode = render_mode
        # Kept as max_steps in the public API for compatibility. It counts
        # internal game frames, while one RL decision advances frame_skip frames.
        self.max_frames = max_steps
        self.frame_skip = 4
        self.current_frame = 0
        self.agent_role = agent_role or Gunner()
        self.opponent_role = opponent_role or Gunner()
        self.opponent_agent = opponent_agent
        self.opponent_bot = opponent_bot

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.OBSERVATION_SIZE,),
            dtype=np.float32,
        )
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
        self._opponent_controller.current_action = 0
        self.current_frame = 0

        px = int(
            self.np_random.integers(
                S.ARENA_LEFT + S.PLAYER_RADIUS,
                S.ARENA_RIGHT - S.PLAYER_RADIUS + 1,
            )
        )
        py = int(
            self.np_random.integers(
                S.ARENA_TOP + S.PLAYER_RADIUS,
                S.ARENA_BOTTOM - S.PLAYER_RADIUS + 1,
            )
        )

        while True:
            ox = int(
                self.np_random.integers(
                    S.ARENA_LEFT + S.PLAYER_RADIUS,
                    S.ARENA_RIGHT - S.PLAYER_RADIUS + 1,
                )
            )
            oy = int(
                self.np_random.integers(
                    S.ARENA_TOP + S.PLAYER_RADIUS,
                    S.ARENA_BOTTOM - S.PLAYER_RADIUS + 1,
                )
            )
            if math.hypot(ox - px, oy - py) > 200:
                break

        self.agent = Player(
            x=px,
            y=py,
            color=S.PURPLE,
            controller=self._rl_controller,
            role=type(self.agent_role)(),
        )

        if self.opponent_agent is not None:
            opponent_controller = self._opponent_controller
        else:
            opponent_controller = self.opponent_bot()

        self.opponent = Player(
            x=ox,
            y=oy,
            color=S.ORANGE,
            controller=opponent_controller,
            role=type(self.opponent_role)(),
        )

        # A fixed right-facing reset made an unconditional right shot a strong
        # and repeatable local optimum.
        self.agent.last_direction = self._random_direction()
        self.opponent.last_direction = self._random_direction()

        self.game = Game(self.agent, self.opponent)

        if self.render_mode == "human" and self._screen is None:
            self._init_pygame()

        return self._get_obs(self.agent, self.opponent), {}

    def _random_direction(self):
        directions = (
            (-1.0, 0.0),
            (1.0, 0.0),
            (0.0, -1.0),
            (0.0, 1.0),
            (-1.0, -1.0),
            (1.0, -1.0),
            (-1.0, 1.0),
            (1.0, 1.0),
        )
        direction = pygame.math.Vector2(
            directions[int(self.np_random.integers(0, len(directions)))]
        )
        return direction.normalize()

    def step(self, action):
        self._rl_controller.current_action = int(action)

        if self.opponent_agent is not None:
            opponent_obs = self._get_obs(self.opponent, self.agent)
            opponent_action = self.opponent_agent.select_action(opponent_obs)
            if isinstance(opponent_action, tuple):
                opponent_action = opponent_action[0]
            self._opponent_controller.current_action = int(opponent_action)

        previous_agent_hp = self.agent.hp
        previous_opponent_hp = self.opponent.hp
        winner = None

        for _ in range(self.frame_skip):
            winner = self.game.step(keys=None, dt=1 / S.FPS)
            self.current_frame += 1

            if self.render_mode == "human":
                self.render()

            if winner is not None or self.current_frame >= self.max_frames:
                break

        damage_dealt = max(0.0, previous_opponent_hp - self.opponent.hp)
        damage_taken = max(0.0, previous_agent_hp - self.agent.hp)

        # A role-neutral reward: trading equal damage is neutral, dealing damage
        # is good, taking damage is bad. No closeness term forces ranged roles
        # into point-blank combat, and reward is calculated once per RL decision.
        reward = (damage_dealt - damage_taken) / S.PLAYER_MAX_HP

        terminated = winner is not None
        truncated = not terminated and self.current_frame >= self.max_frames

        if terminated:
            reward += 1.0 if winner is self.agent else -1.0
        elif truncated:
            reward -= 0.25

        info = {
            "damage_dealt": damage_dealt,
            "damage_taken": damage_taken,
            "agent_hp": self.agent.hp,
            "opponent_hp": self.opponent.hp,
        }
        return (
            self._get_obs(self.agent, self.opponent),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def _get_obs(self, me, them):
        # Absolute self position preserves wall awareness; opponent and hazards
        # are relative to the observing player.
        player_x = 2.0 * (me.x - S.ARENA_LEFT) / self._ARENA_WIDTH - 1.0
        player_y = 2.0 * (me.y - S.ARENA_TOP) / self._ARENA_HEIGHT - 1.0
        opponent_relative_x = np.clip(
            (them.x - me.x) / self._ARENA_WIDTH, -1.0, 1.0
        )
        opponent_relative_y = np.clip(
            (them.y - me.y) / self._ARENA_HEIGHT, -1.0, 1.0
        )

        player_hp = np.clip(me.hp / S.PLAYER_MAX_HP, 0.0, 1.0)
        opponent_hp = np.clip(them.hp / S.PLAYER_MAX_HP, 0.0, 1.0)
        player_cooldown = self._cooldown_fraction(me)
        opponent_cooldown = self._cooldown_fraction(them)

        player_movement = getattr(me, "move_direction", pygame.math.Vector2())
        opponent_movement = getattr(
            them, "move_direction", pygame.math.Vector2()
        )
        distance = np.clip(
            math.hypot(me.x - them.x, me.y - them.y) / self._MAX_DISTANCE,
            0.0,
            1.0,
        )
        opponent_immobile = 1.0 if them.stun_timer > 0 else 0.0

        observation = np.array(
            [
                player_x,
                player_y,
                opponent_relative_x,
                opponent_relative_y,
                player_hp,
                opponent_hp,
                player_cooldown,
                opponent_cooldown,
                me.last_direction.x,
                me.last_direction.y,
                player_movement.x,
                player_movement.y,
                them.last_direction.x,
                them.last_direction.y,
                opponent_movement.x,
                opponent_movement.y,
                distance,
                opponent_immobile,
                *self._role_one_hot(them.role),
                *self._get_hazards(me, them),
            ],
            dtype=np.float32,
        )
        return np.clip(observation, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _cooldown_fraction(player):
        if player.role.cooldown <= 0:
            return 0.0
        return float(np.clip(player.cooldown / player.role.cooldown, 0.0, 1.0))

    def _role_one_hot(self, role):
        values = [0.0] * len(self.ROLE_CLASSES)
        index = self.ROLE_INDEX.get(type(role))
        if index is not None:
            values[index] = 1.0
        return values

    def _get_hazards(self, me, them):
        hazards = []
        for projectile in them.projectiles:
            if not projectile.get("alive", False):
                continue

            relative_x = np.clip(
                (projectile["x"] - me.x) / self._ARENA_WIDTH, -1.0, 1.0
            )
            relative_y = np.clip(
                (projectile["y"] - me.y) / self._ARENA_HEIGHT, -1.0, 1.0
            )
            velocity_x = np.clip(
                projectile.get("dx", 0.0) / self._MAX_HAZARD_SPEED,
                -1.0,
                1.0,
            )
            velocity_y = np.clip(
                projectile.get("dy", 0.0) / self._MAX_HAZARD_SPEED,
                -1.0,
                1.0,
            )
            distance = math.hypot(
                projectile["x"] - me.x, projectile["y"] - me.y
            )
            hazards.append(
                (distance, 1.0, relative_x, relative_y, velocity_x, velocity_y)
            )

        hazards.sort(key=lambda hazard: hazard[0])
        result = []
        for index in range(2):
            if index < len(hazards):
                result.extend(hazards[index][1:])
            else:
                # Presence=0 distinguishes no hazard from a stationary hazard
                # exactly on the observing player.
                result.extend((0.0, 0.0, 0.0, 0.0, 0.0))
        return result

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

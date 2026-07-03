import math
import os
import random
from functools import partial

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv

import settings as S
from entities.player import Player
from game import Game
from systems.controller import EasyBot, RLController
from systems.roles import Blackhole, Bomber, Dasher, Gunner, Splitter, ToxicTrail


def recommended_num_envs(max_workers=16):
    """Use a power-of-two worker count that divides the training budgets."""
    override = os.getenv("ARENA_NUM_ENVS")
    if override:
        return max(1, int(override))
    if hasattr(os, "sched_getaffinity"):
        cpu_count = len(os.sched_getaffinity(0))
    else:
        cpu_count = os.cpu_count() or 1
    available = max(1, min(max_workers, cpu_count))
    workers = 1
    while workers * 2 <= available:
        workers *= 2
    return workers


def create_vector_env(
    agent_role,
    opponent_mix,
    opponent_roles,
    num_envs,
    seed,
    asynchronous=True,
):
    """Create seeded simulation workers for one curriculum phase."""
    env_fns = [
        partial(
            ArenaBrawlEnv,
            agent_role=agent_role(),
            opponent_mix=opponent_mix,
            opponent_roles=opponent_roles,
        )
        for _ in range(num_envs)
    ]
    if asynchronous and num_envs > 1:
        # Spawn is safe after PyTorch has initialized a CUDA context in the
        # learner process; forking a CUDA-initialized process is not.
        env = AsyncVectorEnv(
            env_fns,
            context="spawn",
            autoreset_mode=AutoresetMode.SAME_STEP,
        )
    else:
        env = SyncVectorEnv(
            env_fns, autoreset_mode=AutoresetMode.SAME_STEP
        )
    states, _ = env.reset(seed=[seed + index for index in range(num_envs)])
    return env, states


class ArenaBrawlEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    _ARENA_WIDTH = S.ARENA_RIGHT - S.ARENA_LEFT
    _ARENA_HEIGHT = S.ARENA_BOTTOM - S.ARENA_TOP
    _MAX_DISTANCE = math.hypot(_ARENA_WIDTH, _ARENA_HEIGHT)
    _MAX_HAZARD_SPEED = 600.0

    ROLE_CLASSES = (Gunner, Bomber, Dasher, Blackhole, ToxicTrail, Splitter)
    ROLE_INDEX = {role_class: index for index, role_class in enumerate(ROLE_CLASSES)}

    # Compact state used by every algorithm:
    # self position (2), opponent relative position (2), HP/cooldowns (4),
    # self facing (2), opponent facing/movement (4), opponent role (1),
    # and the nearest hazard (present, x, y, vx, vy) (5) = 20.
    OBSERVATION_SIZE = 20

    def __init__(
        self,
        agent_role=None,
        opponent_role=None,
        render_mode=None,
        max_steps=5400,
        opponent_agent=None,
        opponent_bot=EasyBot,
        opponent_mix=None,
        opponent_roles=None,
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
        self.opponent_mix = opponent_mix
        self.opponent_roles = opponent_roles

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
        if seed is not None:
            random.seed(seed)

        if self.opponent_mix:
            factories = [factory for factory, _ in self.opponent_mix]
            weights = np.asarray(
                [weight for _, weight in self.opponent_mix], dtype=np.float64
            )
            weights /= weights.sum()
            index = int(self.np_random.choice(len(factories), p=weights))
            self.opponent_bot = factories[index]

        if self.opponent_roles:
            role_index = int(self.np_random.integers(0, len(self.opponent_roles)))
            self.opponent_role = self.opponent_roles[role_index]()
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

        opponent_movement = getattr(
            them, "move_direction", pygame.math.Vector2()
        )

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
                them.last_direction.x,
                them.last_direction.y,
                opponent_movement.x,
                opponent_movement.y,
                self._role_value(them.role),
                *self._get_nearest_hazard(me, them),
            ],
            dtype=np.float32,
        )
        return np.clip(observation, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _cooldown_fraction(player):
        if player.role.cooldown <= 0:
            return 0.0
        return float(np.clip(player.cooldown / player.role.cooldown, 0.0, 1.0))

    def _role_value(self, role):
        index = self.ROLE_INDEX.get(type(role))
        if index is None or len(self.ROLE_CLASSES) <= 1:
            return 0.0
        return index / (len(self.ROLE_CLASSES) - 1)

    def _get_nearest_hazard(self, me, them):
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

        if not hazards:
            # Presence=0 distinguishes no hazard from a stationary hazard
            # exactly on the observing player.
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        return min(hazards, key=lambda hazard: hazard[0])[1:]

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

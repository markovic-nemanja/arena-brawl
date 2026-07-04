import pygame
import settings as S
import math
import random

class KeyboardController:
    def __init__(self, up, down, left, right):
        self.up = up
        self.down = down
        self.left = left
        self.right = right

    def get_movement(self, keys, player, opponent):
        dx, dy = 0, 0
        if keys[self.up]: dy-=1
        if keys[self.down]: dy+=1
        if keys[self.left]: dx-=1
        if keys[self.right]: dx+=1

        length = math.hypot(dx, dy)  # normalize so diagonals aren't faster (was sqrt(2) times faster)
        if length > 0:
            dx /= length
            dy /= length

        return dx, dy

    def get_action(self, keys, player, opponent):
        return keys[pygame.K_SPACE]

class StationaryBot:
    """Stays still and never fires.
    Used as first opponent, to teach agent to move and fire without being extremely punished."""
    def get_movement(self, keys, player, opponent):
        return 0, 0
    def get_action(self, keys, player, opponent):
        return False

class RandomBot:
    """Moves randomly
    Used as a simple opponent to teach agent to aim and fire when possible"""
    def __init__(self):
        self.dx, self.dy = 0, 0
        self.timer = 0

    def get_movement(self, keys, player, opponent):
        self.timer -= 1
        if self.timer <= 0:
            angle = random.uniform(0, 2 * math.pi)
            self.dx, self.dy = math.cos(angle), math.sin(angle)
            self.timer = random.randint(30, 60)

        return self.dx, self.dy

    def get_action(self, keys, player, opponent):
        return False

class RandomShooterBot(RandomBot):
    """Random movement bot, but fires when possible"""
    def get_action(self, keys, player, opponent):
        return player.role.should_attack(player, opponent)

class AimShooterBot(RandomBot):
    """Random movement but aims at the opponent when firing for 50% of the time.
    Implemented for teaching agent to dodge."""
    def __init__(self, aim_prob=0.5):
        super().__init__()
        self.aim_prob = aim_prob

    def get_movement(self, keys, player, opponent):
        self.timer -= 1
        if self.timer <= 0:
            if random.random() < self.aim_prob:
                dx = opponent.x - player.x
                dy = opponent.y - player.y
                dist = math.hypot(dx, dy)
                self.dx, self.dy = (dx / dist, dy / dist) if dist else (1.0, 0.0)
            else:
                angle = random.uniform(0, 2 * math.pi)
                self.dx, self.dy = math.cos(angle), math.sin(angle)
            self.timer = random.randint(30, 60)
        return self.dx, self.dy

    def get_action(self, keys, player, opponent):
        return player.role.should_attack(player, opponent)

class EasyBot:
    """Chases opponent and fires when possible.
    Used as phase opponent - goal teach agent to make space"""
    def __init__(self):
        pass

    def get_movement(self, keys, player, opponent):
        dx = opponent.x - player.x
        dy = opponent.y - player.y
        distance = math.hypot(dx, dy)
        if distance == 0:
            return 0, 0
        return dx / distance, dy / distance

    def get_action(self, keys, player, opponent):
        return player.role.should_attack(player, opponent)

class GentleAggressor(EasyBot):
    """Same playstyle as EasyBot, but fires only given % of time.
    Used as phase opponent - between two extreme bots"""
    def __init__(self, fire_prob=0.4):
        self.fire_prob = fire_prob

    def get_action(self, keys, player, opponent):
        return player.role.should_attack(player, opponent) and random.random() < self.fire_prob

class MediumBot:
    """Kites and strafes, retreats at low HP, fires when possible.w"""
    def __init__(self):
        self.kite_timer = 0
        self.strafe_timer = 0
        self.strafing = False
        self.strafe_direction = 1

    def get_movement(self, keys, player, opponent):
        dx = opponent.x - player.x
        dy = opponent.y - player.y
        distance = math.hypot(dx, dy)
        if distance == 0:
            return 0, 0
        dx /= distance
        dy /= distance

        if player.hp < 30:
            return -dx, -dy

        if self.kite_timer > 0:
            self.kite_timer -= 1
            return -dx, -dy

        self.strafe_timer -= 1
        if self.strafe_timer <= 0:
            self.strafing = not self.strafing

            if self.strafing:
                self.strafe_direction *= -1

            self.strafe_timer = random.randint(30, 90)

        if self.strafing:
            return -dy * self.strafe_direction, dx * self.strafe_direction

        return dx, dy

    def get_action(self, keys, player, opponent):
        if player.cooldown <= 0 and player.role.should_attack(player, opponent):
            self.kite_timer = 30
            return True

        return False

class HardBot(MediumBot):
    def __init__(self):
        super().__init__()
        self.dodge_radius = 250   # px
        self.dodge_aim    = 0.7   # cos(~45 degrees): only dodge shots roughly aimed at us
        self.wall_margin  = 70    # px from a wall where the center-bias kicks in
        self.wall_weight  = 1.0   # strength of the push back toward open space

    def get_movement(self, keys, player, opponent):
        dodge = self._incoming_dodge(player, opponent)
        if dodge is not None:
            return self._avoid_walls(player, dodge[0], dodge[1])

        mx, my = super().get_movement(keys, player, opponent)
        return self._avoid_walls(player, mx, my)

    def _incoming_dodge(self, player, opponent):
        best = None
        best_dist = self.dodge_radius
        for p in opponent.projectiles:
            if not p.get("alive"):
                continue

            vx = p.get("dx", 0)
            vy = p.get("dy", 0)
            speed = math.hypot(vx, vy)

            if speed == 0:
                continue

            to_me_x = player.x - p["x"]
            to_me_y = player.y - p["y"]
            dist = math.hypot(to_me_x, to_me_y)

            if dist == 0 or dist > self.dodge_radius:
                continue

            aim_dot = (vx / speed) * (to_me_x / dist) + (vy / speed) * (to_me_y / dist)
            if aim_dot < self.dodge_aim:
                continue

            if dist < best_dist:
                best_dist = dist
                best = (vx/speed, vy/speed, to_me_x, to_me_y)

        if best is None:
            return None

        vhx, vhy, to_me_x, to_me_y = best

        perp_x, perp_y = -vhy, vhx

        if perp_x * to_me_x + perp_y * to_me_y < 0:
            perp_x, perp_y = -perp_x, -perp_y

        return perp_x, perp_y

    def _avoid_walls(self, player, mx, my):
        margin = self.wall_margin
        if player.x < S.ARENA_LEFT + S.PLAYER_RADIUS + margin:
            mx += self.wall_weight

        elif player.x > S.ARENA_RIGHT - S.PLAYER_RADIUS - margin:
            mx -= self.wall_weight

        if player.y < S.ARENA_TOP + S.PLAYER_RADIUS + margin:
            my += self.wall_weight

        elif player.y > S.ARENA_BOTTOM - S.PLAYER_RADIUS - margin:
            my -= self.wall_weight

        length = math.hypot(mx, my)
        if length > 0:
            mx /= length
            my /= length

        return mx, my

class GentleMedium(MediumBot):
    """MediumBot (kites, strafes, retreats at low HP) but fires only fire_prob
    Agent should learn to hunt a target that keeps its distance instead of turtling in a corner."""
    def __init__(self, fire_prob=0.5):
        super().__init__()
        self.fire_prob = fire_prob

    def get_action(self, keys, player, opponent):
        if player.cooldown <= 0 and random.random() < self.fire_prob and player.role.should_attack(player, opponent):
            self.kite_timer = 30
            return True
        return False

_DIAG = math.sqrt(2) / 2
_ACTION_TO_MOVE = [
    (0, 0),          # 0 nothing
    (0, -1),         # 1 up
    (0, 1),          # 2 down
    (-1, 0),         # 3 left
    (1, 0),          # 4 right
    (-_DIAG, -_DIAG),# 5 up-left
    (_DIAG, -_DIAG), # 6 up-right
    (-_DIAG, _DIAG), # 7 down-left
    (_DIAG, _DIAG),  # 8 down-right
    (0, 0)           # 9 ability - get_action will be fired
]

class RLController:
    def __init__(self):
        self.current_action = 0

    def get_movement(self, keys, player, opponent):
        return _ACTION_TO_MOVE[self.current_action]

    def get_action(self, keys, player, opponent):
        return self.current_action == 9

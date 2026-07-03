import pygame
import settings as S
from .base import Role
import math

class Gunner(Role):
    def __init__(self):
        super().__init__(
            name="Gunner",
            color=S.PURPLE,
            cooldown=1.5,
            desc="Fires a spread burst of bullets in a cone",
            select_color=(242, 193, 78),
        )
        self.bullet_speed  = 480
        self.bullet_damage = 7
        self.bullet_radius = 5
        self.pellet_count  = 5          # bullets per burst, fanned across the cone
        self.spread_deg    = 20         # half-angle of the cone (bullets span -20..+20 deg)
        self.attack_range  = 300

    def activate(self, player):
        self._fire_fan(player)

    def update(self, dt, player, opponent):
        for p in player.projectiles:
            if p["type"] != "bullet":
                continue
            p["x"] += p["dx"] * dt
            p["y"] += p["dy"] * dt
            if (p["x"] < S.ARENA_LEFT or p["x"] > S.ARENA_RIGHT or
                    p["y"] < S.ARENA_TOP or p["y"] > S.ARENA_BOTTOM):
                p["alive"] = False

    def _fire_fan(self, player):
        d = pygame.math.Vector2(player.last_direction)
        if d.length() == 0:
            return
        d = d.normalize()
        n = self.pellet_count
        for i in range(n):
            # evenly space the pellets across [-spread, +spread]; single pellet -> straight ahead
            frac = i / (n - 1) if n > 1 else 0.5
            angle = -self.spread_deg + frac * 2 * self.spread_deg
            di = d.rotate(angle)
            player.projectiles.append({
                "x": player.x, "y": player.y,
                "dx": di.x * self.bullet_speed,
                "dy": di.y * self.bullet_speed,
                "type": "bullet",
                "alive": True,
                "damage": self.bullet_damage,
                "radius": self.bullet_radius,
            })

    def draw(self, screen, player):
        for p in player.projectiles:
            if p["alive"] and p["type"] == "bullet":
                pygame.draw.circle(screen, S.ORANGE,
                                   (int(p["x"]), int(p["y"])), p["radius"])

    def get_hazards(self, player):
        return [(p["x"], p["y"]) for p in player.projectiles
                if p["alive"] and p["type"] == "bullet"]

    def should_attack(self, player, opponent):
        dx = opponent.x - player.x
        dy = opponent.y - player.y
        distance = math.hypot(dx, dy)

        if distance == 0:
            return False

        dot = player.last_direction.x * (dx/distance) + player.last_direction.y * (dy/distance)
        return distance < self.attack_range and dot > 0.5

import pygame
import settings as S
from .base import Role


class Bomber(Role):
    def __init__(self):
        super().__init__(
            name="Bomber",
            color=S.ORANGE,
            cooldown=3.0,
            desc="Throws a bomb that explodes in a big AOE",
            select_color=(229, 87, 63),
        )
        self.bomb_fuse = 1.3 # seconds until explosion
        self.bomb_travel_speed = 200 # pixels per second
        self.bomb_radius = 100
        self.bomb_damage = 35
        self.attack_range = self.bomb_travel_speed * self.bomb_fuse

    def activate(self, player):
        d = pygame.math.Vector2(player.last_direction)
        if d.length() == 0:
            d = pygame.math.Vector2(1, 0)
        d = d.normalize()
        player.projectiles.append({
            "x": player.x + d.x * S.PLAYER_RADIUS,
            "y": player.y + d.y * S.PLAYER_RADIUS,
            "dx": d.x * self.bomb_travel_speed,
            "dy": d.y * self.bomb_travel_speed,
            "type": "bomb",
            "timer": self.bomb_fuse,
            "alive": True,
            "damage": self.bomb_damage,
            "radius": self.bomb_radius,
        })

    def update(self, dt, player, opponent):
        for p in player.projectiles:
            if p["type"] != "bomb":
                continue
            p["timer"] -= dt
            if p["timer"] > 0:
                p["x"] += p["dx"] * dt
                p["y"] += p["dy"] * dt
                if p["x"] <= S.ARENA_LEFT or p["x"] >= S.ARENA_RIGHT:
                    p["dx"] *= -1
                if p["y"] <= S.ARENA_TOP or p["y"] >= S.ARENA_BOTTOM:
                    p["dy"] *= -1
            elif p["timer"] <= -0.3:
                p["alive"] = False

    def draw(self, screen, player):
        for p in player.projectiles:
            if not p["alive"] or p["type"] != "bomb":
                continue
            if p["timer"] > 0:
                pygame.draw.circle(screen, self.color,
                                   (int(p["x"]), int(p["y"])), 12)
            else:
                pygame.draw.circle(screen, S.RED,
                                   (int(p["x"]), int(p["y"])), p["radius"], 3)

    def get_hazards(self, player):
        return [(p["x"], p["y"]) for p in player.projectiles
                if p["alive"] and p["type"] == "bomb"]

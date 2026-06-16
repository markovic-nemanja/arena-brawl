import pygame
import settings as S
from .base import Role

class Splitter(Role):
    def __init__(self):
        super().__init__(
            name="Splitter",
            color=S.SPLIT,
            cooldown=1.8,
            desc="Fires a bullet that splits into two if it misses",
            select_color=(245, 215, 90)
        )
        self.bullet_speed = 480
        self.bullet_radius = 6
        self.damage = 12
        self.max_range = 240

    def activate(self, player):
        d = pygame.math.Vector2(player.last_direction).normalize()
        player.projectiles.append({
            "type": "split_bullet",
            "x": player.x,
            "y": player.y,
            "dx": d.x * self.bullet_speed,
            "dy": d.y * self.bullet_speed,
            "can_split": True,
            "traveled": 0,
            "damage": self.damage,
            "radius": self.bullet_radius,
            "alive": True
        })

    def update(self, dt, player, opponent):
        new_bullets = []
        for p in player.projectiles:
            if p["type"] != "split_bullet" or not p["alive"]:
                continue
            p["x"] += p["dx"] * dt
            p["y"] += p["dy"] * dt
            out_of_bounds = (p["x"] < S.ARENA_LEFT or p["x"] > S.ARENA_RIGHT or
                    p["y"] < S.ARENA_TOP or p["y"] > S.ARENA_BOTTOM)
            reached_range = False
            if p["can_split"]:
                p["traveled"] += self.bullet_speed * dt
                reached_range = p["traveled"] >= self.max_range
            if out_of_bounds or reached_range:
                p["alive"] = False
                if p["can_split"]:
                    x = max(S.ARENA_LEFT, min(S.ARENA_RIGHT, p["x"]))
                    y = max(S.ARENA_TOP, min(S.ARENA_BOTTOM, p["y"]))
                    new_bullets.append({
                        "type": "split_bullet",
                        "x": x,
                        "y": y,
                        "dx": -p["dy"],
                        "dy": p["dx"],
                        "can_split": False,
                        "damage": self.damage,
                        "radius": self.bullet_radius,
                        "alive": True
                    })
                    new_bullets.append({
                        "type": "split_bullet",
                        "x": x,
                        "y": y,
                        "dx": p["dy"],
                        "dy": -p["dx"],
                        "can_split": False,
                        "damage": self.damage,
                        "radius": self.bullet_radius,
                        "alive": True
                    })
        player.projectiles.extend(new_bullets)

    def draw(self, screen, player):
        for p in player.projectiles:
            if p["alive"] and p["type"] == "split_bullet":
                pygame.draw.circle(screen, S.SPLIT, (int(p["x"]), int(p["y"])), p["radius"])

    def get_hazards(self, player):
        return [(p["x"], p["y"]) for p in player.projectiles
                if p["alive"] and p["type"] == "split_bullet"]

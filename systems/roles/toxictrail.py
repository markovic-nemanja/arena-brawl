import pygame
import settings as S
from .base import Role
import math

class ToxicTrail(Role):
    def __init__(self):
        super().__init__(
            name="Toxic Trail",
            color=S.TOXIC,
            cooldown=2.5,
            desc="Fires a toxic trail that lingers and damages on contact",
            select_color=(140, 220, 90)
        )
        self.head_speed = 350
        self.head_radius = 30
        self.damage = 10
        self.segment_interval = 0.20
        self.segment_radius = 30
        self.segment_lifetime = 4
        self.max_segments = 6
        self.attack_range = self.head_speed * self.segment_interval * self.max_segments

    def activate(self, player):
        d = pygame.math.Vector2(player.last_direction).normalize()
        player.projectiles.append({
            "type": "toxic_head",
            "x": player.x,
            "y": player.y,
            "dx": d.x * self.head_speed,
            "dy": d.y * self.head_speed,
            "segments_spawned": 0,
            "segment_timer": 0,
            "damage": self.damage,
            "radius": self.head_radius,
            "alive": True
        })

    def update(self, dt, player, opponent):
        new_segments = []
        for p in player.projectiles:
            if not p["alive"]:
                continue
            if p["type"] == "toxic_head":
                p["x"] += p["dx"] * dt
                p["y"] += p["dy"] * dt
                if (p["x"] < S.ARENA_LEFT or p["x"] > S.ARENA_RIGHT or
                        p["y"] < S.ARENA_TOP or p["y"] > S.ARENA_BOTTOM):
                    p["alive"] = False
                    continue
                p["segment_timer"] += dt
                if p["segment_timer"] >= self.segment_interval:
                    p["segment_timer"] = 0
                    new_segments.append({
                        "type": "toxic_trail",
                        "x": p["x"],
                        "y": p["y"],
                        "lifetime": self.segment_lifetime,
                        "damage": self.damage,
                        "radius": self.segment_radius,
                        "alive": True
                    })
                    p["segments_spawned"] += 1
                    if p["segments_spawned"] >= self.max_segments:
                        p["alive"] = False
            elif p["type"] == "toxic_trail":
                p["lifetime"] -= dt
                if p["lifetime"] <= 0:
                    p["alive"] = False
        player.projectiles.extend(new_segments)

    def draw(self, screen, player):
        for p in player.projectiles:
            if not p["alive"]:
                continue
            if p["type"] == "toxic_head":
                pygame.draw.circle(screen, S.TOXIC, (int(p["x"]), int(p["y"])), p["radius"])
            elif p["type"] == "toxic_trail":
                fade = max(0, min(255, int(160 * (p["lifetime"] / self.segment_lifetime))))
                segment = pygame.Surface((p["radius"] * 2, p["radius"] * 2), pygame.SRCALPHA)
                pygame.draw.circle(segment, (*S.TOXIC, fade), (p["radius"], p["radius"]), p["radius"])
                screen.blit(segment, (int(p["x"]) - p["radius"], int(p["y"]) - p["radius"]))

    def get_hazards(self, player):
        return [(p["x"], p["y"]) for p in player.projectiles
                if p["alive"] and p["type"] in ("toxic_head", "toxic_trail")]

    def should_attack(self, player, opponent):
        dx = opponent.x - player.x
        dy = opponent.y - player.y
        distance = math.hypot(dx, dy)
        if distance == 0:
            return False
        # A.x * B.x + A.y * B.y = cos(θ) the dot product of two vectors, where θ is the angle between them
        dot = player.last_direction.x * (dx / distance) + player.last_direction.y * (dy / distance)
        return dot > 0.7 # 0.7 is roughly 45 degrees, so the opponent must be in front of the player
import pygame
import settings as S
from .base import Role


class Gunner(Role):
    def __init__(self):
        super().__init__(
            name="Gunner",
            color=S.PURPLE,
            cooldown=1.5,
            desc="3-bullet burst, direction changes between shots",
            select_color=(242, 193, 78),
        )
        self.burst_queue   = []
        self.burst_elapsed = 0
        self.bullet_speed = 480
        self.bullet_damage = 10
        self.burst_delay = 0.25
        self.bullet_radius = 5

    def activate(self, player):
        self._fire(player)
        self.burst_queue   = [self.burst_delay, self.burst_delay * 2]
        self.burst_elapsed = 0

    def update(self, dt, player, opponent):
        if self.burst_queue:
            self.burst_elapsed += dt
            while self.burst_queue and self.burst_elapsed >= self.burst_queue[0]:
                self._fire(player)
                self.burst_queue.pop(0)

        for p in player.projectiles:
            if p["type"] != "bullet":
                continue
            p["x"] += p["dx"] * dt
            p["y"] += p["dy"] * dt
            if (p["x"] < S.ARENA_LEFT or p["x"] > S.ARENA_RIGHT or
                    p["y"] < S.ARENA_TOP or p["y"] > S.ARENA_BOTTOM):
                p["alive"] = False

    def _fire(self, player):
        d = pygame.math.Vector2(player.last_direction)
        if d.length() == 0:
            return
        d = d.normalize()
        player.projectiles.append({
            "x": player.x, "y": player.y,
            "dx": d.x * self.bullet_speed,
            "dy": d.y * self.bullet_speed,
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

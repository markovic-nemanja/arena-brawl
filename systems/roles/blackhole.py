import pygame
import settings as S
from .base import Role

class Blackhole(Role):
    def __init__(self):
        super().__init__(
            name = "Blackhole",
            color = S.VOID,
            cooldown = 3.0,
            desc = "Places a zone that traps and damages anyone who enters it",
            select_color = S.LAVENDER
        )
        
        self.throw_distance = 150
        self.radius = 55
        self.zone_duration = 6.0
        self.trap_duration = 1.5
        self.damage_per_second = 15
        self.pull_speed = 100
        self.max_active = 4
        
    def activate(self, player):
        d = pygame.math.Vector2(player.last_direction).normalize()
        
        zones = [p for p in player.projectiles if p["type"] == "blackhole" and p["alive"]]
        if len(zones) >= self.max_active:
            zones[0]["alive"] = False
            
        x = player.x + d.x * self.throw_distance
        y = player.y + d.y * self.throw_distance
        x = max(S.ARENA_LEFT + self.radius, min(S.ARENA_RIGHT - self.radius, x))
        y = max(S.ARENA_TOP + self.radius, min(S.ARENA_BOTTOM - self.radius, y))
        
        player.projectiles.append({
            "type": "blackhole",
            "x": x,
            "y": y,
            "phase": "armed",
            "timer": self.zone_duration,
            "trap_duration": self.trap_duration,
            "damage_per_second": self.damage_per_second,
            "pull_speed": self.pull_speed,
            "radius": self.radius,
            "alive": True
        })
        
    def update(self, dt, player, opponent):
        for p in player.projectiles:
            if p["type"] != "blackhole" or not p["alive"]:
                continue
            
            p["timer"] -= dt
            if p["timer"] <= 0:
                p["alive"] = False
                
    def draw(self, screen, player):
        for p in player.projectiles:
            if not p["alive"] or p["type"] != "blackhole":
                continue
            
            color = self.select_color if p["phase"] == "armed" else self.color
            zone = pygame.Surface((p["radius"] * 2, p["radius"] * 2), pygame.SRCALPHA)
            pygame.draw.circle(zone, (*color, 128), (p["radius"], p["radius"]), p["radius"])
            screen.blit(zone, (int(p["x"]) - p["radius"], int(p["y"]) - p["radius"]))
            
    def get_hazards(self, player):
        return [(p["x"], p["y"]) for p in player.projectiles
                if p["alive"] and p["type"] == "blackhole"]
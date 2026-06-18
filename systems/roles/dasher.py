import pygame
import settings as S
from .base import Role
import math

class Dasher(Role):
    def __init__(self):
        super().__init__(
            name="Dasher",
            color=S.CYAN,
            cooldown=1.5,
            desc="Dashes in the last direction, dealing damage to anyone in the way",
            select_color=(0, 255, 255),
        )
        self.dash_distance = 220 #pixels
        self.phantom_time = 1 # seconds
        self.phantom_hitbox = 20 # pixels
        self.phantom_damage = 30
        self.attack_range = self.dash_distance + S.PLAYER_SPEED * 60 * self.phantom_time # 60 because 60 FPS

    def activate(self, player):
        d = pygame.math.Vector2(player.last_direction).normalize()
        origin_x, origin_y = player.x, player.y
        left = S.ARENA_LEFT + S.PLAYER_RADIUS
        right = S.ARENA_RIGHT - S.PLAYER_RADIUS
        top = S.ARENA_TOP + S.PLAYER_RADIUS
        bottom = S.ARENA_BOTTOM - S.PLAYER_RADIUS
        
        player.x = max(left, min(right, player.x + d.x * self.dash_distance))
        player.y = max(top, min(bottom, player.y + d.y * self.dash_distance))
        
        player.projectiles.append({
            "type": "phantom",
            "x": origin_x,
            "y": origin_y,
            "phase" : "waiting",
            "timer": self.phantom_time,
            "target_x": None,
            "target_y": None,
            "alive": True,
            "damage": self.phantom_damage,
            "radius": self.phantom_hitbox
        })
        
    def update(self, dt, player, opponent):
        for p in player.projectiles:
            if p["type"] != "phantom":
                continue
            if p["phase"] == "waiting":
                p["timer"] -= dt
                if p["timer"] <= 0:
                    p["phase"] = "merging"
                    p["target_x"] = player.x
                    p["target_y"] = player.y
                    
    def draw(self, screen, player):
        for p in player.projectiles:
            if not p["alive"] or p["type"] != "phantom":
                continue
            if p["phase"] == "waiting":
                path_color = (*player.color, 128) #transparent follow line
                path = pygame.Surface((S.SCREEN_W, S.SCREEN_H), pygame.SRCALPHA)
                pygame.draw.line(path, path_color, (int(p["x"]), int(p["y"])), (int(player.x), int(player.y)), S.PLAYER_RADIUS * 2)
                pygame.draw.circle(path, path_color, (int(p["x"]), int(p["y"])), S.PLAYER_RADIUS)
                pygame.draw.circle(path, path_color, (int(player.x), int(player.y)), S.PLAYER_RADIUS)
                screen.blit(path, (0,0))
            phantom = pygame.Surface((S.PLAYER_RADIUS * 2, S.PLAYER_RADIUS * 2), pygame.SRCALPHA)
            pygame.draw.circle(phantom, (*player.color, 100), (S.PLAYER_RADIUS, S.PLAYER_RADIUS), S.PLAYER_RADIUS)
            screen.blit(phantom, (int(p["x"]) - S.PLAYER_RADIUS, int(p["y"]) - S.PLAYER_RADIUS))
            
    def get_hazards(self, player):
        return [(p["x"], p["y"]) for p in player.projectiles
                if p["alive"] and p["type"] == "phantom" and p["phase"] == "waiting"]
        
    def should_attack(self, player, opponent):
        dx = opponent.x - player.x
        dy = opponent.y - player.y
        distance = math.hypot(dx, dy)
        if distance == 0:
            return False
        # A.x * B.x + A.y * B.y = cos(θ) the dot product of two vectors, where θ is the angle between them
        dot = player.last_direction.x * (dx / distance) + player.last_direction.y * (dy / distance)
        return distance < self.attack_range and dot > 0.7 # 0.7 is roughly 45 degrees, so the opponent must be in front of the player
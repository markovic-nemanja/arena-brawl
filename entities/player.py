import pygame
import settings as S
from systems import ui

class Player:
    def __init__(self, x, y, color, controller, role):
        self.x = x
        self.y = y
        self.color = color
        self.controller = controller
        self.role = role
        self.hp = S.PLAYER_MAX_HP
        self.projectiles = []
        self.cooldown = 0
        self.vx = 0
        self.vy = 0
        self.last_direction = pygame.math.Vector2(1, 0)  # default facing right

    def use_ability(self):
        if self.cooldown > 0:
            return
        self.role.activate(self)
        self.cooldown = self.role.cooldown

    def update(self, keys, opponent, dt):
        dx, dy = self.controller.get_movement(keys, self, opponent)

        # Track last facing direction for ability aim
        if dx != 0 or dy != 0:
            self.last_direction = pygame.math.Vector2(dx, dy)

        self.vx *= 0.8
        self.vy *= 0.8
        self.x += dx * S.PLAYER_SPEED + self.vx
        self.y += dy * S.PLAYER_SPEED + self.vy

        left = S.ARENA_LEFT + S.PLAYER_RADIUS
        right = S.ARENA_RIGHT - S.PLAYER_RADIUS
        top = S.ARENA_TOP + S.PLAYER_RADIUS
        bottom = S.ARENA_BOTTOM - S.PLAYER_RADIUS

        if self.x <= left:
            self.x  = left
            self.vx = S.BOUNCE_FORCE
        elif self.x >= right:
            self.x  = right
            self.vx = -S.BOUNCE_FORCE

        if self.y <= top:
            self.y  = top
            self.vy = S.BOUNCE_FORCE
        elif self.y >= bottom:
            self.y = bottom
            self.vy = -S.BOUNCE_FORCE

        if self.cooldown > 0:
            self.cooldown -= dt

        # Delegate projectile movement and ability state to role
        self.role.update(dt, self, opponent)

        # Remove dead projectiles
        self.projectiles = [p for p in self.projectiles if p["alive"]]

    def draw(self, screen):
        cx, cy = int(self.x), int(self.y)

        self.role.draw(screen, self)  # Role-specific visuals

        # Soft glow behind player
        glow_r = S.PLAYER_RADIUS + 10
        glow = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow, (*self.color, 70), (glow_r, glow_r), glow_r)
        screen.blit(glow, (cx - glow_r, cy - glow_r))

        pygame.draw.circle(screen, self.color, (cx, cy), S.PLAYER_RADIUS)

        label = ui.font(28).render(str(self.hp), True, S.WHITE)
        screen.blit(label, label.get_rect(center=(cx, cy)))

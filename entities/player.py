import pygame
import settings as S

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
        
    def use_ability(self, target_x, target_y):
        if self.cooldown > 0:
            return
        new_projectiles = self.role.create_projectiles(self.x, self.y, target_x, target_y)
        self.projectiles.extend(new_projectiles)
        self.cooldown = self.role.cooldown

        
    def update(self, keys, opponent, dt):
        dx, dy = self.controller.get_movement(keys, self, opponent)
        
        self.vx *= 0.8
        self.vy *= 0.8
        self.x += dx * S.PLAYER_SPEED + self.vx
        self.y += dy * S.PLAYER_SPEED + self.vy
        
        left   = S.ARENA_MARGIN + S.PLAYER_RADIUS
        right  = S.SCREEN_W - S.ARENA_MARGIN - S.PLAYER_RADIUS
        top    = S.ARENA_MARGIN + S.PLAYER_RADIUS
        bottom = S.SCREEN_H - S.ARENA_MARGIN - S.PLAYER_RADIUS

        if self.x <= left:
            self.x  = left
            self.vx = S.BOUNCE_FORCE        # snap to wall, shoot right
        elif self.x >= right:
            self.x  = right
            self.vx = -S.BOUNCE_FORCE       # snap to wall, shoot left

        if self.y <= top:
            self.y  = top
            self.vy = S.BOUNCE_FORCE        # snap to wall, shoot down
        elif self.y >= bottom:
            self.y  = bottom
            self.vy = -S.BOUNCE_FORCE       # snap to wall, shoot up
        
        if self.cooldown > 0:
            self.cooldown -= dt
            
        for p in self.projectiles:
            if p["type"] == "bullet" and p["timer"] <= 0:
                p["x"] += p["dx"]
                p["y"] += p["dy"]
            p["timer"] -= dt
            if p["type"] == "bomb" and p["timer"] <= -0.3:
                p["alive"] = False
        
    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), S.PLAYER_RADIUS)
        font = pygame.font.SysFont(None, 28)
        label = font.render(str(self.hp), True, S.WHITE)
        screen.blit(label, label.get_rect(center=(int(self.x), int(self.y))))
        
        for p in self.projectiles:
            if not p['alive']:
                continue
            if p['type'] == 'bullet' and p["timer"] <= 0:
                pygame.draw.circle(screen, S.ORANGE, (int(p['x']), int(p['y'])), 5)
            elif p['type'] == 'bomb':
                pygame.draw.circle(screen, self.color, (int(p['x']), int(p['y'])), 10)
            else:
                pygame.draw.circle(screen, S.RED, (int(p['x']), int(p['y'])), S.BOMB_RADIUS, 2)
import pygame
import settings as S

class Player:
    def __init__(self, x, y, color, controller):
        self.x = x
        self.y = y
        self.color = color
        self.controller = controller
        self.hp = S.PLAYER_MAX_HP
        
    def update(self, keys, opponent):
        dx, dy = self.controller.get_movement(keys, self, opponent)
        self.x += dx * S.PLAYER_SPEED
        self.y += dy * S.PLAYER_SPEED
        
        self.x = max(S.ARENA_MARGIN + 28, min(S.SCREEN_W - S.ARENA_MARGIN - 28, self.x))
        self.y = max(S.ARENA_MARGIN + 28, min(S.SCREEN_H - S.ARENA_MARGIN - 28, self.y))
        
    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), 28)
        font = pygame.font.SysFont(None, 28)
        label = font.render(str(self.hp), True, S.WHITE)
        screen.blit(label, label.get_rect(center=(int(self.x), int(self.y))))
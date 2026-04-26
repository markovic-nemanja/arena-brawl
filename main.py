import pygame
import sys
import settings as S
from entities.player import Player
from systems.controller import KeyboardController, AIController
from systems.roles import Gunner, Bomber
from systems import collision

pygame.init()
screen = pygame.display.set_mode((S.SCREEN_W, S.SCREEN_H))
pygame.display.set_caption("ARENA")
clock = pygame.time.Clock()

p1 = Player(x=220, y=360, color=S.PURPLE, controller=KeyboardController(pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d), role=Gunner())
p2 = Player(x=500, y=360, color=S.ORANGE, controller=AIController(), role=Bomber())

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    
    screen.fill(S.BLACK)
    pygame.draw.rect(screen, S.WHITE, (S.ARENA_MARGIN, S.ARENA_MARGIN, S.SCREEN_W - S.ARENA_MARGIN * 2, S.SCREEN_H - S.ARENA_MARGIN * 2), 2)
    dt = clock.tick(S.FPS) / 1000
    keys = pygame.key.get_pressed()
    mouse_x, mouse_y = pygame.mouse.get_pos()
    p1.update(keys, p2, dt)
    p2.update(keys, p1, dt)
    collision.update(p1, p2)
    
    if p1.hp <= 0 or p2.hp <= 0:
        winner = "P2" if p1.hp <= 0 else "P1"
        font = pygame.font.SysFont(None, 72)
        text = font.render(f"{winner} WINS!", True, S.WHITE)
        screen.blit(text, text.get_rect(center=(S.SCREEN_W // 2, S.SCREEN_H // 2)))
        pygame.display.flip()
        pygame.time.wait(3000)
        pygame.quit()
        sys.exit()
    
    if p1.controller.get_action(keys, p1, p2):
        p1.use_ability(mouse_x, mouse_y)
    if p2.controller.get_action(keys, p2, p1):
        p2.use_ability(p1.x, p1.y)
    
    p1.draw(screen)
    p2.draw(screen)
    pygame.display.flip()
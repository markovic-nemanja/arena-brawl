import pygame
import sys
import settings as S
from entities.player import Player
from systems.controller import KeyboardController, AIController

pygame.init()
screen = pygame.display.set_mode((S.SCREEN_W, S.SCREEN_H))
pygame.display.set_caption("ARENA")
clock = pygame.time.Clock()

p1 = Player(x=220, y=360, color=S.PURPLE, controller=KeyboardController(pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d))
p2 = Player(x=500, y=360, color=S.ORANGE, controller=AIController())

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    
    screen.fill(S.BLACK)
    pygame.draw.rect(screen, S.WHITE, (S.ARENA_MARGIN, S.ARENA_MARGIN, S.SCREEN_W - S.ARENA_MARGIN * 2, S.SCREEN_H - S.ARENA_MARGIN * 2), 2)
    keys = pygame.key.get_pressed()
    p1.update(keys, p2)
    p2.update(keys, p1)
    p1.draw(screen)
    p2.draw(screen)
    pygame.display.flip()
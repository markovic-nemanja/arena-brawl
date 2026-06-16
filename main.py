import pygame
import sys
import settings as S
from entities.player import Player
from systems.controller import KeyboardController, AIController
from systems.roles import Gunner, Bomber, ROLES
from systems import ui
from game import Game

pygame.init()
screen = pygame.display.set_mode((S.SCREEN_W, S.SCREEN_H))
pygame.display.set_caption("Arena Brawl")
clock = pygame.time.Clock()


def make_players(mode):
    P1X, P2X, Y = 226, 494, 403
    if mode == 1:
        role1 = ui.role_select(screen, 1, ROLES)
        p1 = Player(x=P1X, y=Y, color=S.PURPLE, controller=KeyboardController(pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d), role=role1)
        p2 = Player(x=P2X, y=Y, color=S.ORANGE, controller=AIController(), role=Bomber())
    elif mode == 2:
        role1 = ui.role_select(screen, 1, ROLES)
        role2 = ui.role_select(screen, 2, ROLES)
        p1 = Player(x=P1X, y=Y, color=S.PURPLE, controller=KeyboardController(pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d), role=role1)
        p2 = Player(x=P2X, y=Y, color=S.ORANGE, controller=KeyboardController(pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT), role=role2)
    else:
        p1 = Player(x=P1X, y=Y, color=S.PURPLE, controller=AIController(), role=Gunner())
        p2 = Player(x=P2X, y=Y, color=S.ORANGE, controller=AIController(), role=Bomber())
    return p1, p2


while True:
    mode = ui.mode_select(screen)
    p1, p2 = make_players(mode)
    game = Game(p1, p2)

    running = True
    while running:
        dt = clock.tick(S.FPS) / 1000
        keys = pygame.key.get_pressed()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        winner = game.step(keys, dt)
        game.render(screen)
        pygame.display.flip()

        if winner:
            ui.draw_game_over(screen, winner.role.name, winner.color)
            pygame.display.flip()
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            pygame.quit()
                            sys.exit()
                        if event.key == pygame.K_r:
                            running = False
                if not running:
                    break

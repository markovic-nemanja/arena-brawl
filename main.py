import pygame
import sys
import settings as S
from entities.player import Player
from systems.controller import KeyboardController, AIController
from systems.roles import Gunner, Bomber, ROLES
from systems import collision, ui

pygame.init()
screen = pygame.display.set_mode((S.SCREEN_W, S.SCREEN_H))
pygame.display.set_caption("Arena Brawl")
clock = pygame.time.Clock()

def mode_select():
    selected = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1: selected = 0
                if event.key == pygame.K_2: selected = 1
                if event.key == pygame.K_3: selected = 2
                if event.key == pygame.K_RETURN:
                    return selected + 1
        
        ui.draw_mode_select(screen, selected)
        pygame.display.flip()
        
def role_select(player_num):
    selected = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT: selected = 0
                if event.key == pygame.K_RIGHT: selected = 1
                if event.key == pygame.K_RETURN:
                    return ROLES[selected]
        
        ui.draw_role_select(screen, selected, player_num)
        pygame.display.flip()
        
def make_players(mode):
    if mode == 1:
        role1 = role_select(1)
        p1 = Player(x=220,y=360, color=S.PURPLE, controller=KeyboardController(pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d), role=role1)
        p2 = Player(x=500,y=360, color=S.ORANGE, controller=AIController(), role=Bomber())
    elif mode == 2:
        role1 = role_select(1)
        role2 = role_select(2)
        p1 = Player(x=220,y=360, color=S.PURPLE, controller=KeyboardController(pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d), role=role1)
        p2 = Player(x=500,y=360, color=S.ORANGE, controller=KeyboardController(pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT), role=role2)
    else:
        p1 = Player(x=220,y=360, color=S.PURPLE, controller=AIController(), role=Gunner())
        p2 = Player(x=500,y=360, color=S.ORANGE, controller=AIController(), role=Bomber())
    return p1, p2

def game_loop(p1,p2):
    while True:
        dt = clock.tick(S.FPS) / 1000
        keys = pygame.key.get_pressed()
        mouse_x, mouse_y = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        p1.update(keys, p2, dt)
        p2.update(keys, p1, dt)
        collision.update(p1, p2)
        
        if p1.controller.get_action(keys, p1, p2):
            p1.use_ability(mouse_x, mouse_y)
        if p2.controller.get_action(keys, p2, p1):
            p2.use_ability(p1.x, p1.y)
            
        ui.draw_arena(screen)
        p1.draw(screen)
        p2.draw(screen)
        ui.draw_hud(screen, p1, p2)
        pygame.display.flip()
        
        if p1.hp <= 0 or p2.hp <= 0:
            winner = "P2" if p1.hp <= 0 else "P1"
            ui.draw_game_over(screen, winner)
            pygame.display.flip()
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            pygame.quit()
                            sys.exit()
                        if event.key == pygame.K_r:
                            waiting = False
                            return
                            
while True:
    mode = mode_select()
    p1, p2 = make_players(mode)
    game_loop(p1, p2)
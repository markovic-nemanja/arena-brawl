import pygame
import settings as S

def draw_arena(screen):
    screen.fill(S.BLACK)
    pygame.draw.rect(screen, S.WHITE, (S.ARENA_MARGIN, S.ARENA_MARGIN, 
                     S.SCREEN_W - S.ARENA_MARGIN * 2, 
                     S.SCREEN_H - S.ARENA_MARGIN * 2), 2)
    
def draw_hud(screen, p1, p2):
    font = pygame.font.SysFont(None, 28)
    
    # P1 health bar - top left
    pygame.draw.rect(screen, S.GRAY,   (20, 20, 200, 18))
    pygame.draw.rect(screen, S.PURPLE, (20, 20, int(200 * p1.hp / S.PLAYER_MAX_HP), 18))
    label = font.render(f"P1  {p1.hp}", True, S.WHITE)
    screen.blit(label, (20, 42))

    # P2 health bar - top right
    pygame.draw.rect(screen, S.GRAY,   (S.SCREEN_W - 220, 20, 200, 18))
    pygame.draw.rect(screen, S.ORANGE, (S.SCREEN_W - 220, 20, int(200 * p2.hp / S.PLAYER_MAX_HP), 18))
    label = font.render(f"P2  {p2.hp}", True, S.WHITE)
    screen.blit(label, (S.SCREEN_W - 220, 42))

def draw_game_over(screen, winner):
    font_big   = pygame.font.SysFont(None, 72)
    font_small = pygame.font.SysFont(None, 36)
    
    overlay = pygame.Surface((S.SCREEN_W, S.SCREEN_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    screen.blit(overlay, (0, 0))
    
    text = font_big.render(f"{winner} WINS!", True, S.WHITE)
    screen.blit(text, text.get_rect(center=(S.SCREEN_W // 2, S.SCREEN_H // 2 - 40)))
    
    hint = font_small.render("Press R to restart  or  Q to quit", True, S.GRAY)
    screen.blit(hint, hint.get_rect(center=(S.SCREEN_W // 2, S.SCREEN_H // 2 + 40)))
    
def draw_mode_select(screen, selected):
    screen.fill(S.BLACK)
    font_big   = pygame.font.SysFont(None, 52)
    font_small = pygame.font.SysFont(None, 32)

    title = font_big.render("Select Game Mode", True, S.WHITE)
    screen.blit(title, title.get_rect(center=(S.SCREEN_W // 2, 160)))

    modes = ["Player vs AI", "1v1 with Friend", "AI vs AI"]
    for i, mode in enumerate(modes):
        color = S.PURPLE if i == selected else S.GRAY
        text  = font_small.render(f"{i + 1}  —  {mode}", True, color)
        screen.blit(text, text.get_rect(center=(S.SCREEN_W // 2, 280 + i * 60)))

    hint = font_small.render("Press Enter to start", True, S.GRAY)
    screen.blit(hint, hint.get_rect(center=(S.SCREEN_W // 2, 520)))
    
def draw_role_select(screen, selected, player_num):
    screen.fill(S.BLACK)
    font_big   = pygame.font.SysFont(None, 52)
    font_small = pygame.font.SysFont(None, 28)
    font_desc  = pygame.font.SysFont(None, 24)

    title = font_big.render(f"Player {player_num} — Choose Role", True, S.WHITE)
    screen.blit(title, title.get_rect(center=(S.SCREEN_W // 2, 100)))

    roles = [
        ("Gunner", S.PURPLE, "Fires a burst of 3 bullets", "Fast cooldown — 1.2s"),
        ("Bomber", S.ORANGE, "Throws 3 bombs in a triangle", "Slow cooldown — 2.5s"),
    ]
    for i, (name, color, desc1, desc2) in enumerate(roles):
        x = 160 + i * 300
        border_color = S.WHITE if i == selected else S.GRAY
        pygame.draw.rect(screen, border_color, (x - 110, 180, 220, 280), 2)
        pygame.draw.circle(screen, color, (x, 280), 40)
        label = font_big.render(name, True, color)
        screen.blit(label, label.get_rect(center=(x, 360)))
        d1 = font_desc.render(desc1, True, S.WHITE)
        d2 = font_desc.render(desc2, True, S.GRAY)
        screen.blit(d1, d1.get_rect(center=(x, 400)))
        screen.blit(d2, d2.get_rect(center=(x, 425)))

    hint = font_small.render("Arrow keys to select  —  Enter to confirm", True, S.GRAY)
    screen.blit(hint, hint.get_rect(center=(S.SCREEN_W // 2, 560)))
    

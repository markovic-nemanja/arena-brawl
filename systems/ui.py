import os
import sys
import pygame
import settings as S

# Bundled fonts (match the design mockup). Space Grotesk for titles/names,
# JetBrains Mono for the small uppercase labels and key hints.
_FONT_DIR  = os.path.join(os.path.dirname(__file__), "..", "assets", "fonts")
_SANS_PATH = os.path.join(_FONT_DIR, "SpaceGrotesk.ttf")
_MONO_PATH = os.path.join(_FONT_DIR, "JetBrainsMono.ttf")

# Fonts are created lazily and cached so we don't rebuild them every frame.
_font_cache = {}
def font(size, bold=False, mono=False):
    key = (size, bold, mono)
    if key not in _font_cache:
        path = _MONO_PATH if mono else _SANS_PATH
        try:
            f = pygame.font.Font(path, size)          # bundled TTF
        except (FileNotFoundError, OSError):
            f = pygame.font.SysFont("Arial", size)    # fallback if files missing
        f.set_bold(bold)                              # synthetic bold from the variable font
        _font_cache[key] = f
    return _font_cache[key]


def _text(screen, txt, size, color, center=None, topleft=None, topright=None,
          bold=False, mono=False):
    surf = font(size, bold, mono).render(txt, True, color)
    rect = surf.get_rect()
    if center:   rect.center = center
    if topleft:  rect.topleft = topleft
    if topright: rect.topright = topright
    screen.blit(surf, rect)
    return rect


# ============================================================
# ARENA
# ============================================================
def draw_arena(screen):
    screen.fill(S.WIN_BG)

    arena = pygame.Rect(S.ARENA_LEFT, S.ARENA_TOP,
                        S.ARENA_RIGHT - S.ARENA_LEFT,
                        S.ARENA_BOTTOM - S.ARENA_TOP)
    pygame.draw.rect(screen, S.ARENA_BG, arena)

    # faint center ring, just for depth
    detail = pygame.Surface((arena.width, arena.height), pygame.SRCALPHA)
    pygame.draw.circle(detail, (255, 255, 255, 16),
                       (arena.width // 2, arena.height // 2), 27, 1)
    screen.blit(detail, (S.ARENA_LEFT, S.ARENA_TOP))

    pygame.draw.rect(screen, S.WHITE, arena, 2)


# ============================================================
# HUD
# ============================================================
def _draw_bar(screen, x, y, w, h, ratio, color, align_right=False):
    """Draw a track + fill bar. ratio is 0..1. Right-aligned fills from the right."""
    pygame.draw.rect(screen, S.BAR_BG, (x, y, w, h), border_radius=3)
    fill_w = max(0, min(w, int(w * ratio)))
    if fill_w > 0:
        fx = x + (w - fill_w) if align_right else x
        pygame.draw.rect(screen, color, (fx, y, fill_w, h), border_radius=3)


def _draw_player_hud(screen, player, right=False):
    bar_w = 300
    x = S.ARENA_RIGHT - bar_w if right else S.ARENA_LEFT
    name_anchor = (S.ARENA_RIGHT, 28) if right else (S.ARENA_LEFT, 28)

    # name
    if right:
        _text(screen, player.role.name.upper(), 22, player.color, topright=name_anchor, bold=True)
    else:
        _text(screen, player.role.name.upper(), 22, player.color, topleft=name_anchor, bold=True)

    # health bar
    hp_ratio = max(0, player.hp) / S.PLAYER_MAX_HP
    _draw_bar(screen, x, 56, bar_w, 18, hp_ratio, player.color, align_right=right)

    # cooldown bar — full + player color when ready, partial + gray while charging
    ready = player.cooldown <= 0
    if ready:
        cd_ratio, cd_color, label = 1.0, player.color, "ABILITY READY"
    else:
        cd_ratio = 1 - (player.cooldown / player.role.cooldown)
        cd_color, label = S.COOL, "COOLING DOWN"
    _draw_bar(screen, x, 81, bar_w, 8, cd_ratio, cd_color, align_right=right)

    if right:
        _text(screen, label, 13, S.TXT_DIMMER, topright=(S.ARENA_RIGHT, 94), mono=True)
    else:
        _text(screen, label, 13, S.TXT_DIMMER, topleft=(S.ARENA_LEFT, 94), mono=True)


def draw_hud(screen, p1, p2):
    _draw_player_hud(screen, p1, right=False)
    _draw_player_hud(screen, p2, right=True)


# ============================================================
# GAME OVER
# ============================================================
def draw_game_over(screen, winner_name, winner_color):
    overlay = pygame.Surface((S.SCREEN_W, S.SCREEN_H), pygame.SRCALPHA)
    overlay.fill((6, 6, 9, 209))
    screen.blit(overlay, (0, 0))

    _text(screen, f"{winner_name} Wins!", 64, winner_color,
          center=(S.SCREEN_W // 2, S.SCREEN_H // 2 - 24), bold=True)
    _text(screen, "R to restart  —  Q to quit", 16, S.TXT_DIM,
          center=(S.SCREEN_W // 2, S.SCREEN_H // 2 + 40), mono=True)


# ============================================================
# MODE SELECT
# ============================================================
def draw_mode_select(screen, selected):
    screen.fill(S.WIN_BG)

    _text(screen, "Arena Brawl", 52, S.TXT, center=(S.SCREEN_W // 2, 150), bold=True)
    _text(screen, "SELECT GAME MODE", 14, S.TXT_DIMMER, center=(S.SCREEN_W // 2, 200), mono=True)

    modes = ["Player vs AI", "1v1 with Friend", "AI vs AI"]
    for i, mode in enumerate(modes):
        sel = (i == selected)
        y = 300 + i * 70
        rect = pygame.Rect(S.SCREEN_W // 2 - 180, y - 26, 360, 52)
        pygame.draw.rect(screen, S.PANEL_SEL if sel else S.PANEL_BG, rect, border_radius=8)
        pygame.draw.rect(screen, S.WHITE if sel else S.LINE_DIM, rect, 2, border_radius=8)
        color = S.TXT if sel else S.TXT_DIM
        _text(screen, f"{i + 1}   {mode}", 26, color, center=(S.SCREEN_W // 2, y))

    _text(screen, "Number keys or arrows to choose  —  Enter to start", 14, S.TXT_DIM,
          center=(S.SCREEN_W // 2, 580), mono=True)


# ============================================================
# CHARACTER SELECT  (5-wide grid, scales to any role count)
# ============================================================
def draw_character_select(screen, selected, player_num, roles):
    screen.fill(S.WIN_BG)

    _text(screen, "Choose Your Character", 30, S.TXT,
          center=(S.SCREEN_W // 2, 50), bold=True)
    _text(screen, f"PLAYER {player_num}  —  {len(roles)} FIGHTERS", 13, S.TXT_DIMMER,
          center=(S.SCREEN_W // 2, 86), mono=True)

    cols = 5
    pad_x = 40
    gap = 13
    top = 130
    card_h = 150
    inner_w = S.SCREEN_W - pad_x * 2
    card_w = (inner_w - gap * (cols - 1)) / cols

    for i, role in enumerate(roles):
        col = i % cols
        row = i // cols
        x = pad_x + col * (card_w + gap)
        y = top + row * (card_h + gap)
        sel = (i == selected)

        rect = pygame.Rect(int(x), int(y), int(card_w), card_h)
        pygame.draw.rect(screen, S.PANEL_SEL if sel else S.PANEL_BG, rect, border_radius=8)
        pygame.draw.rect(screen, S.WHITE if sel else S.LINE_DIM, rect, 2, border_radius=8)

        cx = rect.centerx
        # chip
        if sel:
            pygame.draw.circle(screen, (255, 255, 255), (cx, rect.y + 42), 29, 2)
        pygame.draw.circle(screen, role.select_color, (cx, rect.y + 42), 25)
        # name
        _text(screen, role.name, 17, S.TXT, center=(cx, rect.y + 88), bold=True)
        # ability — wrap to two lines if needed
        _draw_wrapped(screen, role.desc, 13, S.TXT_DIM, cx, rect.y + 108,
                      max_w=int(card_w) - 12)

    _text(screen, "Arrows to navigate  —  Enter to confirm", 14, S.TXT_DIM,
          center=(S.SCREEN_W // 2, S.SCREEN_H - 40), mono=True)


# ============================================================
# SCREEN LOOPS  (input + drawing for menus)
# ============================================================
def mode_select(screen):
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
                if event.key in (pygame.K_UP, pygame.K_LEFT):
                    selected = (selected - 1) % 3
                if event.key in (pygame.K_DOWN, pygame.K_RIGHT):
                    selected = (selected + 1) % 3
                if event.key == pygame.K_RETURN:
                    return selected + 1
        draw_mode_select(screen, selected)
        pygame.display.flip()


def role_select(screen, player_num, roles):
    selected = 0
    cols  = 5
    total = len(roles)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT: selected = (selected + 1) % total
                if event.key == pygame.K_LEFT:  selected = (selected - 1) % total
                if event.key == pygame.K_DOWN:  selected = (selected + cols) % total
                if event.key == pygame.K_UP:    selected = (selected - cols) % total
                if event.key == pygame.K_RETURN:
                    return roles[selected]
        draw_character_select(screen, selected, player_num, roles)
        pygame.display.flip()


def _draw_wrapped(screen, text, size, color, cx, y, max_w):
    """Center-draw text, wrapping onto a second line if it's too wide."""
    f = font(size)
    words = text.split()
    lines, line = [], ""
    for w in words:
        test = (line + " " + w).strip()
        if f.size(test)[0] <= max_w:
            line = test
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)
    for j, ln in enumerate(lines[:2]):
        _text(screen, ln, size, color, center=(cx, y + j * 16))

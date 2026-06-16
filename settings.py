# Screen
SCREEN_W = 720
SCREEN_H = 720
FPS      = 60

# Colors (R, G, B)
WHITE  = (255, 255, 255)
PURPLE = (127, 119, 221)
ORANGE = (216,  90,  48)
RED    = (226,  75,  75)
CYAN   = (86,  204, 196)
VOID = (45, 25, 70)
LAVENDER = (120, 70, 180)
TOXIC = (110, 200, 60)
SPLIT = (235, 200, 60)

# UI theme (from the design mockup)
WIN_BG     = (18, 18, 24)    # game window background
ARENA_BG   = (12, 12, 17)    # inside the arena box
PANEL_BG   = (22, 22, 29)    # character cards
PANEL_SEL  = (29, 29, 38)    # selected card
LINE_DIM   = (51, 51, 61)    # dim borders
TXT        = (236, 236, 237) # main text
TXT_DIM    = (113, 113, 125) # secondary text
TXT_DIMMER = (74, 74, 84)    # faint labels
COOL       = (44, 44, 54)    # cooldown bar (charging)
BAR_BG     = (27, 27, 34)    # empty bar track

# Arena bounds — HUD sits in a band above the arena
ARENA_LEFT   = 24
ARENA_RIGHT  = SCREEN_W - 24      # 696
ARENA_TOP    = 110
ARENA_BOTTOM = SCREEN_H - 24      # 696

# Player / Enemy
PLAYER_SPEED   = 3
PLAYER_MAX_HP  = 100
PLAYER_RADIUS = 38
BOUNCE_FORCE     = 10
STUN_IMMUNITY = 1.0
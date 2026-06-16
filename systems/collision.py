import math
import settings as S

def circles_overlap(ax, ay, ar, bx, by, br):
    distance = math.hypot(ax - bx, ay - by)
    return distance < (ar + br)

def push_apart(player, enemy):
    dist = math.hypot(player.x - enemy.x, player.y - enemy.y)
    min_dist = S.PLAYER_RADIUS * 2
    if dist < min_dist and dist != 0:
        # collision normal - direction from enemy to player
        nx = (player.x - enemy.x) / dist
        ny = (player.y - enemy.y) / dist

        # fix overlap - push them apart
        overlap = min_dist - dist
        player.x += nx * overlap / 2
        player.y += ny * overlap / 2
        enemy.x  -= nx * overlap / 2
        enemy.y  -= ny * overlap / 2

        # relative velocity along collision normal
        dvx = player.vx - enemy.vx
        dvy = player.vy - enemy.vy
        dot = dvx * nx + dvy * ny

        # transfer velocity along collision normal (elastic bounce)
        if dot < 0:
            player.vx -= dot * nx
            player.vy -= dot * ny
            enemy.vx  += dot * nx
            enemy.vy  += dot * ny

        # apply minimum bounce force so every collision feels impactful
        # even if both players were moving slowly toward each other
        player.vx += nx * S.BOUNCE_FORCE
        player.vy += ny * S.BOUNCE_FORCE
        enemy.vx  -= nx * S.BOUNCE_FORCE
        enemy.vy  -= ny * S.BOUNCE_FORCE
        
def check_damage(player, enemy):
    for p in enemy.projectiles:
        if not p["alive"]:
            continue
        if p["type"] == "bullet":
            if circles_overlap(p["x"], p["y"], p["radius"], player.x, player.y, S.PLAYER_RADIUS):
                player.hp -= p["damage"]
                p["alive"] = False
        elif p["type"] == "bomb" and p["timer"] <= 0:
            if circles_overlap(p["x"], p["y"], p["radius"], player.x, player.y, S.PLAYER_RADIUS):
                player.hp -= p["damage"]
                p["alive"] = False
        elif p["type"] == "phantom" and p["phase"] == "merging":
            distance = point_segment_distance(player.x, player.y, p["x"], p["y"], p["target_x"], p["target_y"])
            if distance < p["radius"] + S.PLAYER_RADIUS:
                player.hp -= p["damage"]
            p["alive"] = False

def update(player, enemy):
    push_apart(player, enemy)
    check_damage(player, enemy)
    check_damage(enemy, player)
    
def point_segment_distance(px, py, ax, ay, bx, by):
    abx, aby = bx - ax, by - ay
    squared_length_ab = abx * abx + aby * aby
    if squared_length_ab == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px-ax) * abx + (py-ay) * aby) / squared_length_ab
    t = max(0, min(1, t))
    closest_x = ax + t * abx
    closest_y = ay + t * aby
    return math.hypot(px - closest_x, py - closest_y)
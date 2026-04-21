import settings as S
import math

class Role:
    def __init__(self, name, color, cooldown):
        self.name = name
        self.color = color
        self.cooldown = cooldown
        
    def create_projectiles(self, x, y, target_x, target_y):
        return []
    
class Gunner(Role):
    def __init__(self):
        super().__init__(
            name="Gunner",
            color=S.PURPLE,
            cooldown=0.6
        )
        
    def create_projectiles(self, x, y, target_x, target_y):
        projectiles = []
        angle = math.atan2(target_y - y, target_x - x)
        spreads = [-0.1, 0, 0.1]
        for i, spread in enumerate(spreads):
            projectiles.append({
                "x": x,
                "y": y,
                "dx": math.cos(angle + spread) * S.BULLET_SPEED,
                "dy": math.sin(angle + spread) * S.BULLET_SPEED,
                "type": "bullet",
                "timer": i*S.BURST_DELAY,
                "alive": True
            })
            
        return projectiles
    
class Bomber(Role):
    def __init__(self):
        super().__init__(
            name="Bomber",
            color=S.ORANGE,
            cooldown=1.2
        )
    def create_projectiles(self, x, y, target_x, target_y):
        projectiles = []
        offsets = [(-30, -30), (30,-30), (0, 30)]
        for ox, oy in offsets:
            projectiles.append({
                "x": target_x + ox,
                "y": target_y + oy,
                "type": "bomb",
                "timer": S.BOMB_FUSE,
                "alive": True
            })
            
        return projectiles
    
    
ROLES = [Gunner(), Bomber()]
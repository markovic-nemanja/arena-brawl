import pygame
import settings as S
import math

class KeyboardController:
    def __init__(self, up, down, left, right):
        self.up = up
        self.down = down
        self.left = left
        self.right = right
        
    def get_movement(self, keys, player, opponent):
        dx, dy = 0, 0
        if keys[self.up]: dy-=1
        if keys[self.down]: dy+=1
        if keys[self.left]: dx-=1
        if keys[self.right]: dx+=1
        
        length = math.hypot(dx, dy) # if player is moving diagonally, he move sqrt(2) which is faster than moving straight, so we normalize the vector to length 1
        if length > 0:
            dx /= length
            dy /= length
        
        return dx, dy
    
    def get_action(self, keys, player, opponent):
        return keys[pygame.K_SPACE]
    
class AIController:
    def __init__(self):
        self.attack_range = 300
        
    def get_movement(self, keys, player, opponent):
        dx = opponent.x - player.x
        dy = opponent.y - player.y
        distance = math.hypot(dx, dy)
        if distance == 0:
            return 0, 0
        return dx / distance, dy / distance
    
    def get_action(self, keys, player, opponent):
        distance = math.hypot(opponent.x - player.x, opponent.y - player.y)
        return distance <= self.attack_range
    
_DIAG = math.sqrt(2) / 2
_ACTION_TO_MOVE = [
    (0, 0), # 0 nothing
    (0, -1), # 1 up
    (0, 1), # 2 down
    (-1, 0), # 3 left
    (1, 0), # 4 right
    (-_DIAG, -_DIAG), # 5 up-left
    (_DIAG, -_DIAG), # 6 up-right
    (-_DIAG, _DIAG), # 7 down-left
    (_DIAG, _DIAG),  # 8 down-right
    (0, 0) # 9 ability - get_action will be fired
]

class RLController:
    def __init__(self):
        self.current_action = 0
    
    def get_movement(self, keys, player, opponent):
        return _ACTION_TO_MOVE[self.current_action]
    
    def get_action(self, keys, player, opponent):
        return self.current_action == 9
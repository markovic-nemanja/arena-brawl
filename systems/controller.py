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
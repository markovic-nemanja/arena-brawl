from abc import ABC, abstractmethod
import math

class Role(ABC):
    def __init__(self, name, color, cooldown, desc="", select_color=None):
        self.name         = name
        self.color        = color
        self.cooldown     = cooldown
        self.desc         = desc
        self.select_color = select_color or color

    @abstractmethod
    def activate(self, player):
        """Called once when ability is used."""
        pass

    @abstractmethod
    def update(self, dt, player, opponent):
        """Called every frame — moves projectiles, runs state machines."""
        pass

    @abstractmethod
    def draw(self, screen, player):
        """Draws role-specific visuals."""
        pass

    def get_hazards(self, player):
        """Returns list of (x, y) for active hazards — used by state vector.
        Not abstract — default is no hazards, override when role creates objects."""
        return []

    def should_attack(self, player, opponent):
        """Returns True if the character should attack the opponent.
        Not abstract — default is always attack, override when role has special conditions.
        The default implementation is to attack if the opponent is within attack range, which is defined by the role."""
        return math.hypot(opponent.x - player.x, opponent.y - player.y) <= self.attack_range
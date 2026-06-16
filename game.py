from systems import collision, ui


class Game:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def step(self, keys, dt):
        """Advance one frame. Returns the winner Player or None if still ongoing."""
        self.p1.update(keys, self.p2, dt)
        self.p2.update(keys, self.p1, dt)
        collision.update(self.p1, self.p2, dt)

        if self.p1.controller.get_action(keys, self.p1, self.p2):
            self.p1.use_ability()
        if self.p2.controller.get_action(keys, self.p2, self.p1):
            self.p2.use_ability()

        if self.p1.hp <= 0:
            return self.p2
        if self.p2.hp <= 0:
            return self.p1
        return None

    def render(self, screen):
        """Draw the current frame."""
        ui.draw_arena(screen)
        self.p1.draw(screen)
        self.p2.draw(screen)
        ui.draw_hud(screen, self.p1, self.p2)

import random

try:
    import pygame
except ImportError:
    pygame = None

from neuroevolution.genome import Genome


class Visualizer:
    """Pygame-based visualization of a trained genome navigating to targets."""

    PLAYER_COLOR = (255, 25, 25)
    TARGET_COLOR = (116, 255, 91)
    BG_COLOR = (0, 0, 0)
    RECT_SIZE = 10
    FRAME_DELAY_MS = 50

    def __init__(
        self,
        genome: Genome,
        screen_width: int = 500,
        screen_height: int = 500,
        speed: int = 5,
    ):
        if pygame is None:
            raise ImportError("pygame is required for visualization: pip install pygame")
        self.genome = genome
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.speed = speed

    def _random_position(self, margin: int = 5) -> tuple[int, int]:
        return (
            random.randint(margin, self.screen_width - self.RECT_SIZE),
            random.randint(margin, self.screen_height - self.RECT_SIZE),
        )

    def run(self) -> None:
        """Open a window and animate the genome's pathfinding behaviour."""
        pygame.init()
        win = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Pathfinding Neuroevolution")

        player_x, player_y = self._random_position()
        dest_x, dest_y = self._random_position(margin=20)

        running = True
        while running:
            pygame.time.delay(self.FRAME_DELAY_MS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            diff_x = player_x - dest_x
            diff_y = player_y - dest_y
            output = self.genome.network.feedforward([[diff_x], [diff_y]])

            if output[0][0] > 0.5:
                player_x -= self.speed
            else:
                player_x += self.speed
            if output[1][0] > 0.5:
                player_y -= self.speed
            else:
                player_y += self.speed

            # Reached the target — pick new positions
            if abs(player_x - dest_x) < 10 and abs(player_y - dest_y) < 10:
                player_x, player_y = self._random_position()
                dest_x, dest_y = self._random_position(margin=20)

            # Out of bounds — reset player only
            if not (0 <= player_x <= self.screen_width and 0 <= player_y <= self.screen_height):
                player_x, player_y = self._random_position()

            win.fill(self.BG_COLOR)
            pygame.draw.rect(
                win, self.PLAYER_COLOR,
                pygame.Rect(player_x, player_y, self.RECT_SIZE, self.RECT_SIZE),
            )
            pygame.draw.rect(
                win, self.TARGET_COLOR,
                pygame.Rect(dest_x, dest_y, self.RECT_SIZE, self.RECT_SIZE),
            )
            pygame.display.flip()

        pygame.quit()

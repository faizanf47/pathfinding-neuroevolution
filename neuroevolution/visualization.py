import random
import threading

try:
    import pygame
except ImportError:
    pygame = None

from neuroevolution.config import Config
from neuroevolution.genome import Genome
from neuroevolution.simulation import Simulation
from neuroevolution.snapshot import TrainingSnapshot


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


class TrainingVisualizer:
    """Live visualization of the neuroevolution training process.

    Runs training in a background thread while rendering the current
    best genome's pathfinding behavior and stats in the main thread.
    """

    PLAYER_COLOR = (255, 25, 25)
    TARGET_COLOR = (116, 255, 91)
    BG_COLOR = (0, 0, 0)
    STATS_COLOR = (255, 255, 255)
    RECT_SIZE = 10
    FRAME_DELAY_MS = 50

    def __init__(
        self,
        config: Config | None = None,
        screen_width: int = 500,
        screen_height: int = 500,
        speed: int = 5,
        viz_interval: int = 5,
    ):
        if pygame is None:
            raise ImportError("pygame is required for visualization: pip install pygame")
        self.config = config or Config()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.speed = speed
        self.viz_interval = viz_interval

        self._snapshot: TrainingSnapshot | None = None
        self._current_genome: Genome | None = None
        self._final_genome: Genome | None = None

    def _random_position(self, margin: int = 5) -> tuple[int, int]:
        return (
            random.randint(margin, self.screen_width - self.RECT_SIZE),
            random.randint(margin, self.screen_height - self.RECT_SIZE),
        )

    def _on_generation(
        self,
        generation: int,
        best_genome: Genome,
        avg_fitness: float,
        best_fitness: int,
    ) -> None:
        # Only swap the visualized genome every viz_interval generations
        if generation % self.viz_interval == 0 or generation == 1:
            self._snapshot = TrainingSnapshot(
                generation=generation,
                best_fitness=best_fitness,
                avg_fitness=avg_fitness,
                best_genome=best_genome,
                is_complete=False,
            )
        elif self._snapshot is not None:
            # Update stats without swapping the genome
            self._snapshot = TrainingSnapshot(
                generation=generation,
                best_fitness=best_fitness,
                avg_fitness=avg_fitness,
                best_genome=self._snapshot.best_genome,
                is_complete=False,
            )

    def _training_thread_target(self) -> None:
        sim = Simulation(
            config=self.config,
            screen_width=self.screen_width,
            screen_height=self.screen_height,
            speed=self.speed,
        )
        best_genome, _, _ = sim.run_training(on_generation=self._on_generation)
        self._snapshot = TrainingSnapshot(
            generation=self.config.max_generations,
            best_fitness=best_genome.fitness,
            avg_fitness=0.0,
            best_genome=best_genome,
            is_complete=True,
        )
        self._final_genome = best_genome

    def _draw_stats(
        self,
        win: "pygame.Surface",
        font: "pygame.font.Font",
        snapshot: TrainingSnapshot,
    ) -> None:
        lines = [
            f"Generation: {snapshot.generation} / {self.config.max_generations}",
            f"Best Fitness: {snapshot.best_fitness}",
            f"Avg Fitness:  {snapshot.avg_fitness:.2f}",
        ]
        if snapshot.is_complete:
            lines.append("TRAINING COMPLETE")

        y = 10
        for line in lines:
            surface = font.render(line, True, self.STATS_COLOR)
            win.blit(surface, (10, y))
            y += 20

    def run(self) -> Genome | None:
        """Start live training visualization. Must be called from the main thread.

        Returns the best genome found, or None if closed before training finished.
        """
        pygame.init()
        win = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Pathfinding Neuroevolution — Training")
        font = pygame.font.SysFont("monospace", 16)

        training_thread = threading.Thread(
            target=self._training_thread_target,
            daemon=True,
        )
        training_thread.start()

        player_x, player_y = self._random_position()
        dest_x, dest_y = self._random_position(margin=20)

        running = True
        while running:
            pygame.time.delay(self.FRAME_DELAY_MS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            snapshot = self._snapshot

            win.fill(self.BG_COLOR)

            if snapshot is None:
                surface = font.render("Initializing training...", True, self.STATS_COLOR)
                win.blit(surface, (10, 10))
                pygame.display.flip()
                continue

            # Genome changed — reset player and target positions
            if snapshot.best_genome is not self._current_genome:
                self._current_genome = snapshot.best_genome
                player_x, player_y = self._random_position()
                dest_x, dest_y = self._random_position(margin=20)

            # Run the best genome's network
            diff_x = player_x - dest_x
            diff_y = player_y - dest_y
            output = self._current_genome.network.feedforward([[diff_x], [diff_y]])

            if output[0][0] > 0.5:
                player_x -= self.speed
            else:
                player_x += self.speed
            if output[1][0] > 0.5:
                player_y -= self.speed
            else:
                player_y += self.speed

            # Reached target
            if abs(player_x - dest_x) < 10 and abs(player_y - dest_y) < 10:
                player_x, player_y = self._random_position()
                dest_x, dest_y = self._random_position(margin=20)

            # Out of bounds
            if not (0 <= player_x <= self.screen_width and 0 <= player_y <= self.screen_height):
                player_x, player_y = self._random_position()

            pygame.draw.rect(
                win, self.PLAYER_COLOR,
                pygame.Rect(player_x, player_y, self.RECT_SIZE, self.RECT_SIZE),
            )
            pygame.draw.rect(
                win, self.TARGET_COLOR,
                pygame.Rect(dest_x, dest_y, self.RECT_SIZE, self.RECT_SIZE),
            )
            self._draw_stats(win, font, snapshot)
            pygame.display.flip()

        pygame.quit()
        return self._final_genome

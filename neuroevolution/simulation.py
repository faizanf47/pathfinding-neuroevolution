import random
from collections.abc import Callable

from neuroevolution.config import Config
from neuroevolution.genome import Genome
from neuroevolution.population import Population


class Simulation:
    """Headless simulation for training genomes on the pathfinding task."""

    def __init__(
        self,
        config: Config | None = None,
        screen_width: int = 500,
        screen_height: int = 500,
        speed: int = 5,
        max_steps: int = 1000,
    ):
        self.config = config or Config()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.speed = speed
        self.max_steps = max_steps

    def evaluate_genome(self, genome: Genome) -> int:
        """Run a single pathfinding game and return the fitness score."""
        player_x = random.randint(5, self.screen_width)
        player_y = random.randint(5, self.screen_height)
        dest_x = random.randint(20, self.screen_width)
        dest_y = random.randint(20, self.screen_height)

        fitness = 0
        for _ in range(self.max_steps):
            diff_x = player_x - dest_x
            diff_y = player_y - dest_y

            output = genome.network.feedforward([[diff_x], [diff_y]])

            if output[0][0] > 0.5:
                player_x -= self.speed
            else:
                player_x += self.speed
            if output[1][0] > 0.5:
                player_y -= self.speed
            else:
                player_y += self.speed

            if abs(player_x - dest_x) < 5 and abs(player_y - dest_y) < 5:
                fitness += 1

        return fitness

    def run_training(
        self,
        on_generation: Callable[[int, Genome, float, int], None] | None = None,
    ) -> tuple[Genome, list[float], list[float]]:
        """Train the population over all generations.

        Args:
            on_generation: Optional callback invoked after each generation with
                (generation, best_genome_clone, avg_fitness, best_fitness).

        Returns:
            A tuple of (best_genome, average_fitness_history, best_fitness_history).
        """
        population = Population(self.config)
        population.generate_random()

        best_genome: Genome | None = None
        best_fitness = 0
        avg_fitness_history: list[float] = []
        best_fitness_history: list[float] = []

        for generation in range(1, self.config.max_generations + 1):
            # Evaluate every genome in the population
            for genome in population.genomes:
                genome.fitness = self.evaluate_genome(genome)

            # Track statistics
            gen_fittest = population.find_fittest()
            avg_fitness_history.append(population.average_fitness())
            best_fitness_history.append(gen_fittest.fitness)

            if gen_fittest.fitness > best_fitness:
                best_fitness = gen_fittest.fitness
                best_genome = gen_fittest.clone()

            print(
                f"Generation {generation}: "
                f"best={best_fitness}, avg={population.average_fitness():.2f}"
            )

            if on_generation is not None:
                snapshot_genome = best_genome or gen_fittest
                on_generation(
                    generation,
                    snapshot_genome.clone(),
                    population.average_fitness(),
                    best_fitness,
                )

            population.evolve()

        if best_genome is None:
            best_genome = population.find_fittest()

        return best_genome, avg_fitness_history, best_fitness_history

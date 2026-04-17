from dataclasses import dataclass, field


@dataclass
class Config:
    """Configuration for the neuroevolution algorithm."""

    # Neural network architecture: [input_size, hidden..., output_size]
    network_architecture: list[int] = field(default_factory=lambda: [2, 3, 2])

    # Population
    population_size: int = 100
    max_generations: int = 4000

    # Elitism
    elitism: bool = True
    elite_count: int = 2
    mutate_elite: bool = False

    # Mutation
    mutation_rate: float = 0.015
    mutate_newborn: bool = True
    mutation_version: int = 1  # 1 = random replacement, 2 = perturbation

    # Selection: "roulette_wheel" or "tournament"
    selection_method: str = "roulette_wheel"
    tournament_size: int = 5

    # Crossover
    crossover_version: int = 1  # 1 = uniform, 2 = cut-and-splice
    uniform_rate: float = 0.5

    # Parent selection
    father_always_elite: bool = True
    father_random_elite: bool = True

    # Number of fully random genomes injected each generation
    new_random_per_generation: int = 0

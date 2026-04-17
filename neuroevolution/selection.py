import random

from neuroevolution.genome import Genome


def roulette_wheel_selection(genomes: list[Genome]) -> Genome:
    """Select a genome using fitness-proportionate selection.

    Falls back to random choice when total fitness is zero.
    """
    total_fitness = sum(g.fitness for g in genomes)
    if total_fitness <= 0:
        return random.choice(genomes)

    pick = random.uniform(0, total_fitness)
    current = 0.0
    for genome in genomes:
        current += genome.fitness
        if current > pick:
            return genome
    return genomes[-1]


def tournament_selection(genomes: list[Genome], tournament_size: int) -> Genome:
    """Select the fittest genome from a random tournament subset."""
    competitors = random.sample(genomes, min(tournament_size, len(genomes)))
    return max(competitors, key=lambda g: g.fitness)

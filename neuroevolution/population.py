import math
import random

from neuroevolution.config import Config
from neuroevolution.genome import Genome
from neuroevolution.network import NeuralNetwork
from neuroevolution.selection import roulette_wheel_selection, tournament_selection


class Population:
    """A population of genomes that can evolve over generations."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.genomes: list[Genome] = []

    def generate_random(self) -> None:
        """Fill the population with random genomes."""
        self.genomes = [
            Genome(NeuralNetwork(self.config.network_architecture), self.config)
            for _ in range(self.config.population_size)
        ]

    def find_fittest(self) -> Genome:
        """Return the single fittest genome."""
        return max(self.genomes, key=lambda g: g.fitness)

    def find_top(self, count: int) -> list[Genome]:
        """Return the top ``count`` genomes sorted by fitness descending."""
        return sorted(self.genomes, key=lambda g: g.fitness, reverse=True)[:count]

    def average_fitness(self) -> float:
        if not self.genomes:
            return 0.0
        return sum(g.fitness for g in self.genomes) / len(self.genomes)

    @property
    def size(self) -> int:
        return len(self.genomes)

    def evolve(self) -> None:
        """Create the next generation via selection, crossover, and mutation."""
        elite: list[Genome] = []
        if self.config.elitism:
            elite = self.find_top(self.config.elite_count)
            if self.config.mutate_elite:
                for genome in elite:
                    genome.mutate(self.config.mutation_rate)

        offspring: list[Genome] = []
        offspring_needed = self.size - len(elite) - self.config.new_random_per_generation

        for _ in range(offspring_needed):
            father = self._select_father(elite)
            mother = self._select_parent()
            child = self._crossover(father, mother)
            if self.config.mutate_newborn:
                child.mutate(self.config.mutation_rate)
            offspring.append(child)

        for _ in range(self.config.new_random_per_generation):
            genome = Genome(NeuralNetwork(self.config.network_architecture), self.config)
            if self.config.mutate_newborn:
                genome.mutate(self.config.mutation_rate)
            offspring.append(genome)

        self.genomes = elite + offspring

    # -- Private helpers -----------------------------------------------------

    def _select_father(self, elite: list[Genome]) -> Genome:
        if self.config.father_always_elite and elite:
            if self.config.father_random_elite:
                return random.choice(elite)
            return elite[0]
        return self._select_parent()

    def _select_parent(self) -> Genome:
        if self.config.selection_method == "tournament":
            return tournament_selection(self.genomes, self.config.tournament_size)
        return roulette_wheel_selection(self.genomes)

    def _crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        if self.config.crossover_version == 1:
            return self._uniform_crossover(parent1, parent2)
        return self._cut_crossover(parent1, parent2)

    def _uniform_crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        child = Genome(config=self.config)
        for i in range(parent1.size()):
            if random.random() <= self.config.uniform_rate:
                child.genes[i] = parent1.genes[i]
            else:
                child.genes[i] = parent2.genes[i]
        child.network.from_genes(child.genes)
        return child

    def _cut_crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        child = Genome(config=self.config)
        genes1 = parent1.genes[:]
        genes2 = parent2.genes[:]

        if random.random() < 0.5:
            genes1, genes2 = genes2, genes1

        num_weights = parent1.network.num_weights

        final_genes = genes1[:num_weights]

        biases1 = genes1[num_weights:]
        biases2 = genes2[num_weights:]
        cut = int(math.floor(len(biases1) * random.random()))
        final_genes.extend(biases1[:cut])
        final_genes.extend(biases2[cut:])

        child.genes = final_genes
        child.network.from_genes(child.genes)
        return child

import random

from neuroevolution.config import Config
from neuroevolution.network import NeuralNetwork


class Genome:
    """A genome wrapping a neural network with evolvable genes."""

    def __init__(
        self,
        network: NeuralNetwork | None = None,
        config: Config | None = None,
    ):
        self.config = config or Config()
        self.network = network or NeuralNetwork(self.config.network_architecture)
        self.genes = self.network.to_genes()
        self.fitness: int = 0

    def clone(self) -> "Genome":
        cloned = Genome(config=self.config)
        cloned.genes = self.genes[:]
        cloned.network.from_genes(cloned.genes)
        cloned.fitness = self.fitness
        return cloned

    def size(self) -> int:
        return len(self.genes)

    def mutate(self, mutation_rate: float) -> None:
        if self.config.mutation_version == 1:
            self._mutate_v1(mutation_rate)
        else:
            self._mutate_v2(mutation_rate)

    def _mutate_v1(self, mutation_rate: float) -> None:
        """Replace genes with a random value with probability ``mutation_rate``."""
        for i in range(len(self.genes)):
            if random.random() <= mutation_rate:
                self.genes[i] = random.random()
        self.network.from_genes(self.genes)

    def _mutate_v2(self, mutation_rate: float) -> None:
        """Perturb genes by a scaled random offset."""
        for i in range(len(self.genes)):
            if random.random() <= mutation_rate:
                self.genes[i] += (
                    self.genes[i] * (random.random() - 0.5) * 3
                    + (random.random() - 0.5)
                )
        self.network.from_genes(self.genes)

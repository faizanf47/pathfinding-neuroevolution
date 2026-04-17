import numpy as np

from neuroevolution.config import Config
from neuroevolution.genome import Genome


def test_mutation_updates_network():
    """After mutation, the network weights must reflect the new genes."""
    config = Config(mutation_rate=1.0)
    genome = Genome(config=config)
    weights_before = [w.copy() for w in genome.network.weights]

    genome.mutate(1.0)

    changed = any(
        not np.array_equal(w1, w2)
        for w1, w2 in zip(weights_before, genome.network.weights)
    )
    assert changed


def test_clone_is_independent():
    genome = Genome()
    genome.fitness = 42
    clone = genome.clone()

    assert clone.fitness == 42
    assert clone.genes == genome.genes
    assert clone is not genome

    clone.genes[0] = 999
    assert genome.genes[0] != 999

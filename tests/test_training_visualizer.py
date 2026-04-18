from neuroevolution.config import Config
from neuroevolution.genome import Genome
from neuroevolution.simulation import Simulation


def test_on_generation_callback_is_invoked():
    config = Config(population_size=10, max_generations=3)
    sim = Simulation(config)
    calls: list[tuple[int, Genome, float, int]] = []

    def callback(gen, genome, avg, best):
        calls.append((gen, genome, avg, best))

    sim.run_training(on_generation=callback)

    assert len(calls) == 3
    assert calls[0][0] == 1
    assert calls[1][0] == 2
    assert calls[2][0] == 3


def test_callback_receives_cloned_genome():
    """The genome passed to the callback should be independent of the population."""
    config = Config(population_size=10, max_generations=2)
    sim = Simulation(config)
    genomes: list[Genome] = []

    def callback(gen, genome, avg, best):
        genomes.append(genome)

    sim.run_training(on_generation=callback)

    assert len(genomes) == 2
    # Each callback should receive a distinct genome object
    assert genomes[0] is not genomes[1]


def test_headless_still_works_without_callback():
    config = Config(population_size=10, max_generations=2)
    sim = Simulation(config)
    best, avg_hist, best_hist = sim.run_training()

    assert best is not None
    assert len(avg_hist) == 2
    assert len(best_hist) == 2

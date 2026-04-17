from neuroevolution.config import Config
from neuroevolution.population import Population


def test_generate_random_size():
    config = Config(population_size=10)
    pop = Population(config)
    pop.generate_random()
    assert pop.size == 10


def test_evolve_preserves_size():
    config = Config(population_size=20, elite_count=2)
    pop = Population(config)
    pop.generate_random()

    for g in pop.genomes:
        g.fitness = 1

    pop.evolve()
    assert pop.size == 20


def test_find_fittest():
    config = Config(population_size=5)
    pop = Population(config)
    pop.generate_random()

    pop.genomes[2].fitness = 100
    assert pop.find_fittest() is pop.genomes[2]

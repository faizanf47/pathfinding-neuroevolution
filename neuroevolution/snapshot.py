from dataclasses import dataclass

from neuroevolution.genome import Genome


@dataclass(frozen=True)
class TrainingSnapshot:
    """Immutable snapshot of training state, shared between threads.

    The training thread creates a new instance each generation and
    atomically swaps the reference. The viz thread reads the latest
    reference — no lock needed under CPython's GIL.
    """

    generation: int
    best_fitness: int
    avg_fitness: float
    best_genome: Genome
    is_complete: bool

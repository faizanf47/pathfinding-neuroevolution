import argparse

from neuroevolution import Config, Simulation, TrainingVisualizer, Visualizer


def main():
    parser = argparse.ArgumentParser(description="Pathfinding Neuroevolution")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Watch training in real-time instead of training headlessly first",
    )
    parser.add_argument(
        "--viz-interval",
        type=int,
        default=5,
        help="In live mode, swap the displayed genome every N generations (default: 5)",
    )
    args = parser.parse_args()

    config = Config()

    if args.live:
        viz = TrainingVisualizer(config, viz_interval=args.viz_interval)
        best = viz.run()
        if best is not None:
            print(f"\nTraining complete. Best fitness: {best.fitness}")
        else:
            print("\nVisualization closed before training completed.")
    else:
        sim = Simulation(config)
        fittest, avg_history, best_history = sim.run_training()
        print(f"\nTraining complete. Best fitness: {fittest.fitness}")
        print("Displaying fittest individual...")
        viz = Visualizer(fittest)
        viz.run()


if __name__ == "__main__":
    main()

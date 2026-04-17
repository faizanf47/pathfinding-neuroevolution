from neuroevolution import Config, Simulation, Visualizer


def main():
    config = Config()
    sim = Simulation(config)
    fittest, avg_history, best_history = sim.run_training()

    print(f"\nTraining complete. Best fitness: {fittest.fitness}")
    print("Displaying fittest individual...")

    viz = Visualizer(fittest)
    viz.run()


if __name__ == "__main__":
    main()

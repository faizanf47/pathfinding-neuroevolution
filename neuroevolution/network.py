import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class NeuralNetwork:
    """Feedforward neural network with configurable layer sizes."""

    def __init__(
        self,
        layer_sizes: list[int],
        weights: list[np.ndarray] | None = None,
        biases: list[np.ndarray] | None = None,
    ):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)

        self.weights = (
            weights
            if weights is not None
            else [
                np.random.randn(y, x)
                for x, y in zip(layer_sizes[:-1], layer_sizes[1:])
            ]
        )
        self.biases = (
            biases
            if biases is not None
            else [np.random.randn(y, 1) for y in layer_sizes[1:]]
        )

        self.num_weights = sum(
            layer_sizes[i] * layer_sizes[i - 1]
            for i in range(1, len(layer_sizes))
        )
        self.num_biases = sum(layer_sizes[1:])

    def feedforward(self, a) -> np.ndarray:
        """Return the network output for input ``a`` (shape: (n_inputs, 1))."""
        for b, w in zip(self.biases, self.weights):
            a = _sigmoid(np.dot(w, a) + b)
        return a

    def to_genes(self) -> list[float]:
        """Flatten weights then biases into a 1-D gene list."""
        genes: list[float] = []
        for layer_idx in range(self.num_layers - 1):
            for neuron in range(self.layer_sizes[layer_idx + 1]):
                genes.extend(self.weights[layer_idx][neuron])
        for layer_biases in self.biases:
            for bias in layer_biases:
                genes.extend(bias)
        return genes

    def from_genes(self, genes: list[float]) -> None:
        """Reconstruct weights and biases from a flat gene list."""
        weight_genes = genes[: self.num_weights]
        bias_genes = genes[self.num_weights:]

        idx = 0
        weights = []
        for layer_idx in range(self.num_layers - 1):
            rows = self.layer_sizes[layer_idx + 1]
            cols = self.layer_sizes[layer_idx]
            layer = np.zeros((rows, cols), dtype=np.float64)
            for r in range(rows):
                for c in range(cols):
                    layer[r, c] = weight_genes[idx]
                    idx += 1
            weights.append(layer)

        idx = 0
        biases = []
        for layer_idx in range(1, self.num_layers):
            size = self.layer_sizes[layer_idx]
            layer_bias = np.zeros((size, 1), dtype=np.float64)
            for r in range(size):
                layer_bias[r, 0] = bias_genes[idx]
                idx += 1
            biases.append(layer_bias)

        self.weights = weights
        self.biases = biases

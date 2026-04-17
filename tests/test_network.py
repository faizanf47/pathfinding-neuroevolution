import numpy as np

from neuroevolution.network import NeuralNetwork


def test_feedforward_output_shape():
    net = NeuralNetwork([2, 3, 2])
    output = net.feedforward([[1], [2]])
    assert output.shape == (2, 1)


def test_feedforward_output_bounded():
    """Sigmoid activation keeps outputs in (0, 1)."""
    net = NeuralNetwork([2, 3, 2])
    output = net.feedforward([[100], [-100]])
    assert np.all(output > 0) and np.all(output < 1)


def test_genes_roundtrip():
    """to_genes -> from_genes should reproduce the same network."""
    net = NeuralNetwork([2, 3, 2])
    genes = net.to_genes()

    net2 = NeuralNetwork([2, 3, 2])
    net2.from_genes(genes)

    for w1, w2 in zip(net.weights, net2.weights):
        np.testing.assert_array_equal(w1, w2)
    for b1, b2 in zip(net.biases, net2.biases):
        np.testing.assert_array_equal(b1, b2)


def test_gene_count():
    net = NeuralNetwork([2, 3, 2])
    genes = net.to_genes()
    assert len(genes) == net.num_weights + net.num_biases

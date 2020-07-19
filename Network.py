# Follow-along code from http://neuralnetworksanddeeplearning.com
# Renamed variables for clarity, added comments, and converted to python3

import numpy as np
from random import shuffle


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


class Network:
    def __init__(self, layers):
        self.layers = layers
        self.n_layers = len(layers)
        # Init biases for neurons in layers 2->n
        self.biases = [np.random.randn(x, 1) for x in layers[1:]]
        # Init weights for connections in layers 1->2, 2->3, ..., n-1->n
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]

    def feedforward(self, cascading_input):
        """Compute the output of each layer in order and use it as the input for the next layer"""
        for bias, weight in zip(self.biases, self.weights):
            cascading_input = sigmoid(np.dot(weight, cascading_input) + bias)
        return cascading_input

    def backprop(self, inputs, desired):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        # Init output arrays with zeros so they can be filled by-index later
        nabla_b = [np.zeros(layer.shape) for layer in self.biases]
        nabla_w = [np.zeros(layer.shape) for layer in self.weights]

        # Feedforward through the network, storing all the activations and zs
        # for each layer
        cur_activation = inputs
        activations = [inputs]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, cur_activation) + b
            zs.append(z)
            cur_activation = sigmoid(z)
            activations.append(cur_activation)

        # Calculate nabla_b and nabla_w for the last layer
        delta = (activations[-1] - desired) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Step backwards through the layers and calculate nabla_b and nabla_w for each
        for l in range(2, self.n_layers):
            z = zs[-l]
            sig_prime = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sig_prime
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [
            w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)
        ]

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [
            (np.argmax(self.feedforward(input_)), expected)
            for (input_, expected) in test_data
        ]
        return np.count_nonzero(int(x == y) for (x, y) in test_results)

    def train(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        for epoch in range(epochs):
            shuffle(training_data)
            mini_batches = [
                training_data[i] for i in range(0, len(training_data), mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Epoch {epoch}: {self.evaluate(test_data)} / {len(test_data)}")
            else:
                print(f"Epoch {epoch} complete")


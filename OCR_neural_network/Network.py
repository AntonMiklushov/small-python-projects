import numpy as np


class Network:
    def __init__(self, sizes: list[int]):
        self.cache = None
        self.properties = sizes
        weights = [np.random.randn(sizes[i], sizes[i - 1]) for i in range(1, len(sizes))]
        biases = [np.random.randn(sizes[i], 1) for i in range(1, len(sizes))]
        self.weights_biases = list(zip(weights, biases))

    def forward_propagation(self, initial_input):
        self.cache = {"A0": initial_input}
        current_input = initial_input
        for i, (weights, biases) in enumerate(self.weights_biases):
            z = weights @ current_input + biases
            a = Network.activation_function(z)
            self.cache[f"Z{i + 1}"] = z
            self.cache[f"A{i + 1}"] = a
            current_input = a
        output = Network.softmax(current_input)
        return output

    def backward_propagation(self, predictions, real):
        """
        Computes the gradients of weights and biases for backpropagation.

        Parameters:
        - predictions: Output of the forward propagation (model's predictions).
        - real: Ground truth values (one-hot encoded).

        Returns:
        - gradients: Dictionary containing dW (gradients for weights) and db (gradients for biases).
        """
        gradients = {}

        # Number of samples
        m = real.shape[1]

        # Compute the derivative of the loss with respect to the output layer's activation
        dZ = predictions - real

        # Backpropagate through each layer
        for i in reversed(range(len(self.weights_biases))):
            A_prev = self.cache[f"A{i}"] if i > 0 else self.cache["A0"]

            # Compute gradients for weights and biases
            dW = (1 / m) * dZ @ A_prev.T
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            # Save gradients
            gradients[f"dW{i + 1}"] = dW
            gradients[f"db{i + 1}"] = db

            if i > 0:
                # Backpropagate the error to the previous layer
                W = self.weights_biases[i][0]
                dZ = (W.T @ dZ) * Network.activation_function_derivative(self.cache[f"Z{i}"])

        return gradients

    def update_parameters(self, gradients, learning_rate):
        for i in range(len(self.weights_biases)):
            self.weights_biases[i] = (self.weights_biases[i][0] - learning_rate * gradients[f"dW{i + 1}"],
                                               self.weights_biases[i][1] - learning_rate * gradients[f"db{i + 1}"])

    def save_weights_biases(self, file_path):
        """
        Saves the weights and biases to a text file.

        Parameters:
        - file_path: Path to the file where weights and biases will be saved.
        """
        with open(file_path, 'w') as file:
            for weights, biases in self.weights_biases:
                np.savetxt(file, weights, header=f"Weights: {weights.shape[0]} {weights.shape[1]}", comments="")
                np.savetxt(file, biases, header=f"Biases: {biases.shape[0]} {biases.shape[1]}", comments="")

    def load_weights_biases(self, file_path):
        """
        Loads the weights and biases from a text file.

        Parameters:
        - file_path: Path to the file where weights and biases are stored.
        """
        import os

        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return  # Do nothing if the file does not exist or is empty

        weights_biases = []
        with open(file_path, 'r') as file:
            while True:
                try:
                    # Read weights
                    header = file.readline()
                    if not header.startswith("Weights"):
                        break
                    weights_shape = tuple(map(int, header.split(':')[1].strip().split()))
                    weights = np.loadtxt(file, max_rows=weights_shape[0]).reshape(weights_shape)

                    # Read biases
                    header = file.readline()
                    biases_shape = tuple(map(int, header.split(':')[1].strip().split()))
                    biases = np.loadtxt(file, max_rows=biases_shape[0]).reshape(biases_shape)

                    weights_biases.append((weights, biases))
                except (ValueError, IndexError):
                    break
        self.weights_biases = weights_biases

    @staticmethod
    def activation_function(m):
        return np.tanh(m)

    @staticmethod
    def activation_function_derivative(m):
        return 1 / np.cosh(np.clip(m, -200, 200)) ** 2

    @staticmethod
    def loss_function(out, real):
        return Network.categorical_cross_entropy(out, real)

    @staticmethod
    def categorical_cross_entropy(predictions, real):
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        return -np.sum(real * np.log(predictions)) / predictions.shape[0]

    @staticmethod
    def softmax(ls):
        exp_ls = np.exp(ls - np.max(ls, axis=0, keepdims=True))
        return (exp_ls / np.sum(exp_ls, axis=0, keepdims=True)).squeeze()

    @staticmethod
    def forward_propagation_function(m, wb):
        return wb[0] @ Network.activation_function(m) + wb[1]

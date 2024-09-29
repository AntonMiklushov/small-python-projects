import numpy as np
from math import pi, e
from functools import reduce

I = np.eye(2)

H = np.array([[1, 1], [1, -1]]) / 2 ** 0.5  # Hadamard

X = np.array([[0, 1], [1, 0]])  # Pauli-X

Y = np.array([[0, complex(0, -1)], [complex(0, 1), 0]])  # Pauli-Y

Z = np.array([[1, 0], [0, -1]])  # Pauli-Y

S = np.array([[1, 0], [0, complex(0, 1)]])  # Phase

T = np.array([[1, 0], [0, e ** (complex(0, 1) * pi / 4)]])

CX = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
])

CZ = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1]
])

SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

CCNOT = np.eye(8)
CCNOT[-1][-1] = 0
CCNOT[-2][-2] = 0
CCNOT[-1][-2] = 1
CCNOT[-2][-1] = 1


class QSystem:

    def __init__(self, qubits: int):
        self.size = qubits
        self.state = np.zeros(2 ** qubits, dtype=np.complex128)
        self.state[0] = 1

    def get_probabilities(self):
        probabilities = np.abs(self.state) ** 2
        return [f"|{i:0{str(self.size)}b}>: {probabilities[i]:.4f}" for i in range(len(probabilities))]

    def get_state(self):
        return self.state

    # applies given gate to the given qubit
    def apply(self, gate, *n):
        def calculate_and_apply_operator(g, n, p):
            """
            :param p: number of qubits the gate takes
            """
            O = reduce(np.kron, [I] * n[0] + [g] + [I] * (self.size - n[-1] - p + 1))
            self.state = np.dot(O, self.state)

        def swap(n0, n1):
            for i in range(min(n0, n1), max(n0, n1) - 1):
                calculate_and_apply_operator(SWAP, (i, i + 1), 2)

        match len(n):
            case 1:
                O = reduce(np.kron, [I] * n[0] + [gate] + [I] * (self.size - n[0] - 1))
                self.state = np.dot(O, self.state)
            case 2:
                if abs(n[0] - n[1]) > 1:
                    swap(n[0], n[1] - 1)
                    calculate_and_apply_operator(gate, (n[1] - 1, n[1]), 2)
                    swap(n[0], n[1] - 1)
                    return
                calculate_and_apply_operator(gate, n, 2)
            case 3:
                if abs(n[0] - n[1] == 1) and abs(n[1] - n[2] == 1):
                    calculate_and_apply_operator(gate, n, 3)
                    return

    def measure(self):
        probabilities = np.abs(self.state) ** 2
        outcome = np.random.choice(range(len(self.state)), p=probabilities)
        measured_state = np.zeros(len(self.state), dtype=np.complex128)
        measured_state[outcome] = 1
        return measured_state, outcome

    def __repr__(self):
        return f"|{self.measure()[1]:0{str(self.size)}b}>"

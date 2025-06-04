import numpy as np
import pennylane as qml

from quantumproject.quantum.simulations import von_neumann_entropy


def test_bell_entropy():
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.state()

    state = circuit()
    S = von_neumann_entropy(state, [0])
    assert np.allclose(S, np.log(2), atol=1e-6)

# qgnn.py
# Quantum Graph Neural Network embedding using Qiskit

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliFeatureMap
from qiskit_aer import AerSimulator


def build_qgnn_circuit(features, num_qubits=12, reps=2, entanglement_layers=2):
    """
    Construct QGNN circuit using Pauli Feature Map and entanglement layers.
    """

    qc = QuantumCircuit(num_qubits)

    feature_map = PauliFeatureMap(
        feature_dimension=num_qubits,
        reps=reps,
        entanglement="linear"
    )

    qc.compose(feature_map.assign_parameters(features), inplace=True)

    # Graph-based entanglement layers
    for _ in range(entanglement_layers):
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)

    return qc


def generate_embeddings(X, config):
    """
    Generate QGNN embeddings for dataset.
    """

    num_qubits = config["qgnn"]["num_qubits"]
    reps = config["qgnn"]["feature_map_repetitions"]
    layers = config["qgnn"]["entanglement_layers"]

    simulator = AerSimulator(method="statevector")

    embeddings = []

    for sample in X:

        qc = build_qgnn_circuit(sample, num_qubits, reps, layers)

        qc.save_statevector()

        result = simulator.run(qc).result()

        statevector = result.get_statevector()

        embedding = np.real(np.mean(statevector))

        embeddings.append([embedding])

        # Print circuit information (for reproducibility)
        print("Circuit Depth:", qc.depth())
        print("Gate Count:", qc.count_ops())

    return np.array(embeddings)
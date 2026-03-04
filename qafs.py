# qafs.py
# Quantum-Aware Feature Selection (QAFS)

import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph


def compute_laplacian_scores(X, k=10):
    """
    Compute Laplacian scores for each feature.
    Lower score = better feature preserving local structure.
    """

    # Construct k-nearest neighbor graph
    A = kneighbors_graph(X, k, mode='connectivity', include_self=True)

    # Graph Laplacian
    L = csgraph.laplacian(A, normed=True)

    scores = []

    for i in range(X.shape[1]):
        feature_vector = X[:, i]
        score = feature_vector.T @ L @ feature_vector
        scores.append(score)

    return np.array(scores)


def select_top_features(X, top_k=18, k_neighbors=10):
    """
    Select top features using Laplacian Score.
    """

    scores = compute_laplacian_scores(X, k_neighbors)

    ranked_indices = np.argsort(scores)

    selected = ranked_indices[:top_k]

    return selected


def quantum_fidelity(feature_a, feature_b):
    """
    Approximate quantum fidelity between two feature vectors.
    """

    a = feature_a / np.linalg.norm(feature_a)
    b = feature_b / np.linalg.norm(feature_b)

    fidelity = np.abs(np.dot(a, b)) ** 2

    return fidelity


def fidelity_pruning(X, feature_indices, threshold=0.90):
    """
    Remove redundant features using fidelity threshold.
    """

    selected = list(feature_indices)

    i = 0

    while i < len(selected):
        j = i + 1

        while j < len(selected):

            f1 = X[:, selected[i]]
            f2 = X[:, selected[j]]

            if quantum_fidelity(f1, f2) >= threshold:
                selected.pop(j)
            else:
                j += 1

        i += 1

    return selected[:12]


def run_qafs(X, config):
    """
    Full QAFS pipeline.
    """

    top_features = select_top_features(
        X,
        top_k=config["qafs"]["laplacian_top_features"],
        k_neighbors=config["qafs"]["k_neighbors"]
    )

    final_features = fidelity_pruning(
        X,
        top_features,
        threshold=config["qafs"]["fidelity_threshold"]
    )

    return final_features
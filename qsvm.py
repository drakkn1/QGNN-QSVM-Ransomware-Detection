# qsvm.py
# Quantum Support Vector Machine (QSVM) kernel implementation

import numpy as np


def compute_quantum_kernel(X_train, X_test=None):
    """
    Compute quantum kernel matrix using inner product
    of quantum embeddings.

    If X_test is None, a training kernel matrix is computed.
    """

    if X_test is None:
        X_test = X_train

    n_test = X_test.shape[0]
    n_train = X_train.shape[0]

    kernel_matrix = np.zeros((n_test, n_train))

    for i in range(n_test):
        for j in range(n_train):
            kernel_matrix[i, j] = np.dot(X_test[i], X_train[j])

    return kernel_matrix


def train_qsvm(train_embeddings, labels, C=1.0):
    """
    Train QSVM using a precomputed kernel matrix.
    """

    from sklearn.svm import SVC

    kernel_matrix = compute_quantum_kernel(train_embeddings)

    model = SVC(kernel="precomputed", C=C)

    model.fit(kernel_matrix, labels)

    return model, kernel_matrix


def evaluate_qsvm(model, train_embeddings, test_embeddings, test_labels):
    """
    Evaluate QSVM classifier.
    """

    test_kernel = compute_quantum_kernel(train_embeddings, test_embeddings)

    accuracy = model.score(test_kernel, test_labels)

    return accuracy
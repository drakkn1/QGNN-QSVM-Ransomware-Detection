# main.py
# Main pipeline for the QGNN–QSVM ransomware detection framework

import numpy as np
import pandas as pd
import yaml
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# -----------------------------
# Load configuration
# -----------------------------

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

seed = config["random_seed"]

np.random.seed(seed)
random.seed(seed)

# -----------------------------
# Load dataset
# -----------------------------
# Replace 'dataset.csv' with the actual dataset file

data = pd.read_csv("dataset.csv")

X = data.drop("label", axis=1).values
y = data["label"].values

# -----------------------------
# Train / Test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=config["dataset"]["test_split"],
    stratify=y,
    random_state=seed
)

# -----------------------------
# Feature normalization
# -----------------------------

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Placeholder for QAFS
# -----------------------------
# In the full implementation this should call qafs.py

selected_features = list(range(12))  # example placeholder

X_train = X_train[:, selected_features]
X_test = X_test[:, selected_features]

# -----------------------------
# Placeholder for QGNN embedding
# -----------------------------
# In the full implementation this should call qgnn.py

train_embeddings = X_train
test_embeddings = X_test

# -----------------------------
# QSVM classification
# -----------------------------

clf = SVC(kernel="rbf", C=config["qsvm"]["regularization_C"])

clf.fit(train_embeddings, y_train)

accuracy = clf.score(test_embeddings, y_test)

print("Test Accuracy:", accuracy)
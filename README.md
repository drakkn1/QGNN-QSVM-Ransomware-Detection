# QGNN–QSVM: Quantum Graph Learning Framework for Ransomware Detection

This repository contains the implementation of the hybrid **Quantum Graph Neural Network – Quantum Support Vector Machine (QGNN–QSVM)** framework proposed for ransomware detection using static Portable Executable (PE) structural features.

The framework integrates classical preprocessing, quantum-aware feature selection, graph-based relational modelling, and quantum kernel classification. The goal of this work is to investigate whether quantum-inspired feature embeddings combined with relational feature modelling can improve ransomware detection performance under controlled simulation settings.

---

## Overview

Modern ransomware detection methods often rely on static analysis of executable files. However, structural features extracted from Portable Executable (PE) headers often exhibit complex interdependencies that are difficult to capture using conventional machine learning models.

The proposed framework addresses this challenge through three main components:

1. **Quantum-Aware Feature Selection (QAFS)**  
   Laplacian Score is used to preserve local structure in the feature space, followed by quantum fidelity-based redundancy pruning to select a compact set of quantum-compatible features.

2. **Quantum Graph Neural Network (QGNN)**  
   The selected features are represented as nodes in a correlation-driven graph. Feature dependencies define the entanglement topology of the quantum circuit, allowing relational feature interactions to be captured through quantum state embedding.

3. **Quantum Support Vector Machine (QSVM)**  
   A quantum kernel based on Pauli feature maps is used for classification, enabling nonlinear separation in a high-dimensional Hilbert space.

---

## Framework Architecture

The complete pipeline consists of the following stages:

1. Dataset preprocessing and normalization  
2. Feature alignment across datasets  
3. Quantum-Aware Feature Selection (QAFS)  
4. Correlation-driven feature graph construction  
5. Quantum Graph Neural Network embedding  
6. Quantum kernel computation  
7. QSVM classification

---

## Datasets

Experiments were conducted using the following publicly available datasets:

**1. Ransomware Detection Dataset (GitHub)**  
Static PE structural feature dataset containing ransomware and benign samples.

**2. EMBER Malware Benchmark Dataset**  
Large-scale benchmark dataset containing labeled PE malware samples with standardized feature extraction.

Feature alignment was performed to ensure a common structural feature representation across both datasets.

---

## Quantum Circuit Configuration

The quantum components of the framework were implemented using **Qiskit** with classical statevector simulation.

Key configuration parameters:

- Number of qubits: **12**
- Feature encoding: **Pauli Feature Map**
- Feature map repetitions: **2**
- Entanglement layers: **2**
- Measurement basis: **Pauli-Z**
- Simulation backend: **Qiskit Aer Statevector Simulator**

The entanglement topology of the circuit is determined by the correlation graph constructed from the selected features.

---

## Reproducibility

All experiments were executed using fixed random seeds to ensure deterministic results.

The repository includes:

- Dataset preprocessing scripts
- Feature normalization procedures
- Hyperparameter configuration files
- Quantum circuit definitions
- Training and evaluation scripts

These resources allow the full experimental pipeline to be reproduced under identical settings.

**Repository Structure**

The repository contains the following main files:

main.py – main training and evaluation pipeline
qafs.py – implementation of quantum-aware feature selection
qgnn.py – quantum graph neural network embedding module
qsvm.py – quantum kernel and QSVM classifier
config.yaml – hyperparameter configuration file
requirements.txt – Python dependency list
README.md – project documentation

---

## Repository Structure

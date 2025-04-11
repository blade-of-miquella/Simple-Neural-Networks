import numpy as np


def activation(vector: np.ndarray) -> np.ndarray:
    return np.where(vector > 0, 1, np.where(vector < 0, -1, 0))


def train_bam(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.dot(X.T, Y)


def recall(X: np.ndarray, weights: np.ndarray, direction: str) -> np.ndarray:
    if direction == "to_Y":
        S = np.dot(X, weights)
        return activation(S)
    elif direction == "to_X":
        S = np.dot(X, weights.T)
        return activation(S)

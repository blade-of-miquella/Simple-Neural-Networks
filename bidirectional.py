import numpy as np

def bipolar(vector: np.ndarray) -> np.ndarray:
    return 2 * vector - 1

def binary(vector: np.ndarray) -> np.ndarray:
    return np.where(vector >= 0, 1, 0)

def train_bam(dataset: np.ndarray, n: int = 20, m: int = 3) -> np.ndarray:
    dataset = np.array(dataset)
    X = dataset[:, :n]
    Y = dataset[:, -m:]
    X_bipolar = bipolar(X)
    Y_bipolar = bipolar(Y)
    weights = np.dot(X_bipolar.T, Y_bipolar)
    return weights

def recall(input_vector: np.ndarray, weights: np.ndarray, direction: str) -> np.ndarray:
    if direction == "to_Y":
        s = np.dot(bipolar(input_vector), weights)
        result = binary(s)
    elif direction == "to_X":
        s = np.dot(bipolar(input_vector), weights.T)
        result = binary(s)
    return result

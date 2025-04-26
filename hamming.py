import numpy as np


def sign_activation(s: float) -> int:
    return 1 if s > 0 else -1 if s < 0 else 0


def compute_initial_activation(prototypes: np.ndarray, x: np.ndarray) -> np.ndarray:
    n = x.shape[0]                   
    m = prototypes.shape[0]          
    T = n / 2.0                     
    y0 = np.zeros(m)
    for j in range(m):
        y0[j] = np.dot(prototypes[j] / 2.0, x) + T
    return y0


def asynchronous_relaxation(y0: np.ndarray, e: float, max_iterations: int = 50) -> np.ndarray:
    m = len(y0)
    y = y0.copy()
    for _ in range(max_iterations):
        prev_y = y.copy()
        indices = np.random.permutation(m)
        for j in indices:
            s_j = y[j] - e * (np.sum(y) - y[j])
            y[j] = sign_activation(s_j)
        if np.array_equal(y, prev_y):
            break
    return y


def hamming_network(prototypes: np.ndarray, x: np.ndarray, e: float = 0.02, max_iterations: int = 50) -> int:
    prototypes = np.array(prototypes, dtype=float)
    x = np.array(x, dtype=float)
    y0 = compute_initial_activation(prototypes, x)
    y_final = asynchronous_relaxation(y0, e, max_iterations)
    indices_ones = np.where(y_final == 1)[0]
    if len(indices_ones) == 1:
        return indices_ones[0]
    elif len(indices_ones) > 1:
        return indices_ones[np.argmax(y0[indices_ones])]
    else:
        return int(np.argmax(y_final))
    
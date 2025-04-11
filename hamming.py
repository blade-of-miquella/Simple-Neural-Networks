import numpy as np


def F(S: np.ndarray) -> np.ndarray:
    return np.maximum(0, S)


def hamming_similarity(y_j: np.ndarray, x: np.ndarray) -> int:
    return np.sum(y_j == x)


def maxnet_sync(z0: np.ndarray, e: float, T: int = 4, tol: float = 1e-1) -> np.ndarray:
    m = len(z0)
    z = np.array(z0, dtype=float)
    for t in range(T):
        S = np.array([z[j] - e * (np.sum(z) - z[j]) for j in range(m)], dtype=float)
        z_new = F(S)
        if np.linalg.norm(z_new - z) < tol:
            z = z_new
            break
        z = z_new
    return z


def hamming_network(Y: np.ndarray, x: np.ndarray, e: float = 0.1, T: int = 4) -> int:
    x = np.array(x)
    Y = np.array(Y)
    m = Y.shape[0]
    z0 = np.array([hamming_similarity(Y[j], x) for j in range(m)], dtype=float)
    zT = maxnet_sync(z0, e=e, T=T)
    winner = np.argmax(zT)
    return winner

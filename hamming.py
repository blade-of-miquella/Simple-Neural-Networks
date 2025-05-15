import numpy as np

def compute_initial_activation(prototypes: np.ndarray, x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    m = prototypes.shape[0]
    T = n / 2.0
    y0 = np.zeros(m)
    
    for j in range(m):
        dot = np.dot(prototypes[j], x)
        y0[j] = 0.5 * dot + T
    return y0

def asynchronous_relaxation(y0: np.ndarray, e: float, max_iterations: int = 200) -> np.ndarray:
    m = len(y0)
    y = y0.copy()
    
    for _ in range(max_iterations):
        prev_y = y.copy()
        indices = np.random.permutation(m)
        for j in indices:
            inhibition = np.sum(prev_y) - prev_y[j]
            s_j = prev_y[j] - e * inhibition
            y[j] = s_j if s_j > 0 else 0.0
        if np.allclose(y, prev_y, atol=1e-6):
            break
    return y

def hamming_network(prototypes: np.ndarray, x: np.ndarray, e: float = 0.3, max_iterations: int = 200) -> int:
    prototypes = np.array(prototypes, dtype=float)
    x = np.array(x, dtype=float)
    
    y0 = compute_initial_activation(prototypes, x)
    z = asynchronous_relaxation(y0, e, max_iterations)
    winner = int(np.argmax(z))
    return winner

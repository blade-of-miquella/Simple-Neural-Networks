import numpy as np

def trashhold_func(val: int) -> int:
    return 1 if val >= 0 else 0

def train_hopfield(patterns) -> np.ndarray:
    patterns = np.array(patterns)
    n = patterns.shape[1]
    W = np.zeros((n, n))
    for pattern in patterns:
        bipolar_pattern = 2 * pattern - 1
        for i in range(n):
            for j in range(n):
                if i != j:
                    W[i, j] += bipolar_pattern[i] * bipolar_pattern[j]
    return W

def hopefieldAsync(W: np.ndarray, distorted: np.ndarray) -> np.ndarray:
    updated = distorted.copy()
    for _ in range(200):
        prev = updated.copy()
        for i in range(len(updated)):
            total = np.dot(2 * updated - 1, W[:, i])
            updated[i] = trashhold_func(total)
        if np.array_equal(prev, updated):
            break
    return updated

def hopefieldSync(W: np.ndarray, distorted: np.ndarray) -> np.ndarray:
    updated = distorted.copy()
    for _ in range(200):
        bipolar_input = 2 * updated - 1
        s = np.dot(bipolar_input, W)
        updated = np.array([trashhold_func(val) for val in s])
    return updated

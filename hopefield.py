import numpy as np


def trashhold_func(bit: int) -> int:
    return 1 if bit >= 0 else 0


def hopefieldAsync(vector: np.ndarray, corr: np.ndarray) -> np.ndarray:
    corrupted = corr.copy()
    I = np.eye(len(vector))
    W = np.outer((2 * vector - 1), (2 * vector - 1)) - I
    iter = 0
    while not np.array_equal(vector, corrupted):
        temp = corrupted.copy()
        iter += 1
        if iter > 3 and np.array_equal(temp, corrupted):
            return corrupted
        for i in range(len(corrupted)):
            corrupted[i] = trashhold_func(np.dot(corrupted, W[:, i]))
    return corrupted


def hopefieldSync(vector: np.ndarray, corr: np.ndarray) -> np.ndarray:
    I = np.eye(len(vector))
    W = np.outer((2 * vector - 1), (2 * vector - 1)) - I
    updated_vector = corr.copy()
    for _ in range(5):
        updated_vector = np.dot(updated_vector, W)
        updated_vector = np.array([trashhold_func(val) for val in updated_vector])
    return updated_vector

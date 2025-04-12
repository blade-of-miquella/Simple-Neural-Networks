import numpy as np
import hopefield as hp
import bidirectional as bam
import hamming as hm

vectors = [
    np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]),
    np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
    np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]),
]

corr_count: int = 1000

def corrupt_vector(vector: np.ndarray, flip_probability=0.3) -> np.ndarray:
    corrupted = vector.copy()
    for i in range(len(corrupted)):
        if np.random.rand() < flip_probability:
            corrupted[i] = 1 - corrupted[i]
    return corrupted

def test_combined(templates: int) -> None:
    for i, vector in enumerate(vectors):
        x = vector[:10]
        y = vector[10:]
        weights_bam = bam.train_bam(x.reshape(1, -1), y.reshape(1, -1))
        success_hopfield = 0
        success_bam_x = 0
        success_bam_y = 0
        success_hamming = 0

        for _ in range(templates):
            corrupted_vector = corrupt_vector(vector)

            result_hopfield = hp.hopefieldAsync(vector, corrupted_vector)
            if np.array_equal(result_hopfield, vector):
                success_hopfield += 1

            corrupted_x = corrupted_vector[:10]
            corrupted_y = corrupted_vector[10:]
            restored_y = bam.recall(
                corrupted_x.reshape(1, -1), weights_bam, direction="to_Y"
            ).flatten()
            restored_x = bam.recall(
                corrupted_y.reshape(1, -1), weights_bam, direction="to_X"
            ).flatten()
            if np.array_equal(restored_y, y):
                success_bam_y += 1
            if np.array_equal(restored_x, x):
                success_bam_x += 1

            winner = hm.hamming_network(vectors, corrupted_vector)
            if winner == i:
                success_hamming += 1

        hopfield_rate = (success_hopfield / templates) * 100
        bam_x_rate = (success_bam_x / templates) * 100
        bam_y_rate = (success_bam_y / templates) * 100
        hamming_rate = (success_hamming / templates) * 100

        print(f"\nVector: {vector}")
        print(f"Hopefield Success Rate: {round(hopfield_rate, 2)}% ({success_hopfield}/{templates})")
        print(f"BAM X->Y Success Rate: {round(bam_y_rate, 2)}% ({success_bam_y}/{templates})")
        print(f"BAM Y->X Success Rate: {round(bam_x_rate, 2)}% ({success_bam_x}/{templates})")
        print(f"Hamming Network Success Rate: {round(hamming_rate, 2)}% ({success_hamming}/{templates})")


test_combined(corr_count)
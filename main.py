import numpy as np
import hopefield as hp
import bidirectional as bam
import hamming as hm
import random

vectors = [
    np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]),
    np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
    np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]),
]

corr_count = 100

def corrupt_vector(vector: np.ndarray, corr_bit: int) -> np.ndarray:
    corrupted = vector.copy()
    already_corrupt = []
    for _ in range(corr_bit):
        index = random.randint(0, len(vector) - 1)
        while index in already_corrupt:
            index = random.randint(0, len(vector) - 1)
        corrupted[index] = 1 - corrupted[index]
        already_corrupt.append(index)
    return corrupted

def test_combined(templates: int) -> None:
    test_number = 0
    general_percent = 100
    print("# Test | Hamming   | HopfieldSync | HopfieldAsync | BamX | BamY")
    print("--------------------------------------------------------------")
    while test_number < 20:
        total_success = {"hamming": 0, "hopfield_sync": 0, "hopfield_async": 0, "bam_x": 0, "bam_y": 0}
        for i, vector in enumerate(vectors):
            x = vector[:10]
            y = vector[10:]
            weights_bam = bam.train_bam(x.reshape(1, -1), y.reshape(1, -1))
            for _ in range(templates):
                corrupted_vector = corrupt_vector(vector, test_number)
                result_hopfield_async = hp.hopefieldAsync(vector, corrupted_vector)
                if np.array_equal(result_hopfield_async, vector):
                    total_success["hopfield_async"] += 1
                result_hopfield_sync = hp.hopefieldSync(vector, corrupted_vector)
                if np.array_equal(result_hopfield_sync, vector):
                    total_success["hopfield_sync"] += 1
                corrupted_x = corrupted_vector[:10]
                corrupted_y = corrupted_vector[10:]
                restored_y = bam.recall(corrupted_x.reshape(1, -1), weights_bam, direction="to_Y").flatten()
                restored_x = bam.recall(corrupted_y.reshape(1, -1), weights_bam, direction="to_X").flatten()
                if np.array_equal(restored_y, y):
                    total_success["bam_y"] += 1
                if np.array_equal(restored_x, x):
                    total_success["bam_x"] += 1
                winner = hm.hamming_network(vectors, corrupted_vector)
                if winner == i:
                    total_success["hamming"] += 1
        hamming_result = f"{round(total_success['hamming'] / templates)}/{len(vectors)}"
        hopfield_sync_result = f"{round(total_success['hopfield_sync'] / templates)}/{len(vectors)}"
        hopfield_async_result = f"{round(total_success['hopfield_async'] / templates)}/{len(vectors)}"
        bam_x_result = f"{round(total_success['bam_x'] / templates)}/{len(vectors)}"
        bam_y_result = f"{round(total_success['bam_y'] / templates)}/{len(vectors)}"
        print(
            f"{test_number + 1:<6} | {hamming_result:<9} | {hopfield_sync_result:<12} | {hopfield_async_result:<13} | {bam_x_result:<5} | {bam_y_result:<5}"
        )
        test_number += 1

test_combined(corr_count)

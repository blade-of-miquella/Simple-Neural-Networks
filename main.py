import numpy as np
import hopefield as hp
import bidirectional as bam
import hamming as hm
import random
from typing import List

vectors = [
    np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]),
    np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]),
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

def format_results(success_list : List, templates : int) -> str:
    percents = [round(100 * s / templates) for s in success_list]
    overall = sum(1 for p in percents if p > 70)
    return "|".join(f"{p}%" for p in percents) + f"|{overall}/{len(success_list)}"

def test_combined(templates: int) -> None:
    header = f"#T  | {'Hamming':<25} | {'HopSync':<25} | {'HopAsync':<25} | {'BamX':<25} | {'BamY':<25}"
    print(header)
    print("-" * len(header))
    for test_number in range(20):
        success = {
            "hamming": [0] * len(vectors),
            "hopfield_sync": [0] * len(vectors),
            "hopfield_async": [0] * len(vectors),
            "bam_x": [0] * len(vectors),
            "bam_y": [0] * len(vectors)
        }
        for idx, vector in enumerate(vectors):
            x = vector
            y = vector[-3:]
            weights_bam = bam.train_bam(x.reshape(1, -1), y.reshape(1, -1))
            for _ in range(templates):
                corrupted = corrupt_vector(vector, test_number)
                if np.array_equal(hp.hopefieldSync(vector, corrupted), vector):
                    success["hopfield_sync"][idx] += 1
                if np.array_equal(hp.hopefieldAsync(vector, corrupted), vector):
                    success["hopfield_async"][idx] += 1
                corrupted_x = corrupted
                corrupted_y = corrupted[-3:]
                if np.array_equal(bam.recall(corrupted_x.reshape(1, -1), weights_bam, direction="to_Y").flatten(), y):
                    success["bam_y"][idx] += 1
                if np.array_equal(bam.recall(corrupted_y.reshape(1, -1), weights_bam, direction="to_X").flatten(), x):
                    success["bam_x"][idx] += 1
                if hm.hamming_network(vectors, corrupted) == idx:
                    success["hamming"][idx] += 1

        hamming_result  = format_results(success["hamming"], templates)
        hop_sync_result = format_results(success["hopfield_sync"], templates)
        hop_async_result = format_results(success["hopfield_async"], templates)
        bam_x_result    = format_results(success["bam_x"], templates)
        bam_y_result    = format_results(success["bam_y"], templates)
        
        print(f"{test_number+1:<3} | {hamming_result:<25} | {hop_sync_result:<25} | {hop_async_result:<25} | {bam_x_result:<25} | {bam_y_result:<25}")

test_combined(corr_count)

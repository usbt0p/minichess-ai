import time
import torch
from src.models.dataloaders import MinichessFfnDataset, get_dataloaders


def test_correctness():
    print("=== Testing Correctness ===")
    file_path = "data/training_data_sample.txt"

    # We will load just a few items to check their contents
    dataset = MinichessFfnDataset(file_path)
    print(f"Total samples loaded: {len(dataset)}")

    if len(dataset) == 0:
        print("No data found!")
        return

    features, move, result, score, mask = dataset[0]
    print(f"Features shape: {features.shape} (Expected: [325])")
    print(f"Move idx: {move.item()} (Expected in [0, 599])")
    print(f"Result: {result.item()} (Expected -1.0, 0.0, or 1.0)")
    print(f"Score: {score.item()} (Can be any integer)")

    assert features.shape == (325,), "Features shape mismatch!"
    assert 0 <= move.item() < 600, "Move index out of bounds!"
    assert result.item() in [-1.0, 0.0, 1.0], "Result out of bounds!"

    # check if the first instance of the dataset correctly encoded:
    # fen rnb1k/1p1qp/pPp2/P1PP1/RNBQK w - - 0 4
    # move d1e2
    # score 272
    # ply 6
    # result 0
    # e

    # first the move. d1e2 = 3*24 + 9-1, so 80
    assert move.item() == 80

    # TODO check correctness of feature vector

    print("Correctness checks passed!\n")


def test_performance():
    print("=== Testing Performance ===")
    file_path = "data/training_data_sample.txt"

    # Benchmark Dataset creation (reading and parsing the file)
    start_time = time.time()
    dataset = MinichessFfnDataset(file_path)
    creation_time = time.time() - start_time
    print(f"Dataset parsing time: {creation_time:.4f}s for {len(dataset)} samples")
    print(f"Parsing speed: {len(dataset)/creation_time:.2f} samples/s")

    # Benchmark DataLoader iteration
    batch_size = 512
    train_loader, _ = get_dataloaders(dataset, batch_size=batch_size, num_workers=0)

    start_time = time.time()
    num_batches = 0
    num_samples = 0
    for features, moves, results, score, masks in train_loader:
        num_batches += 1
        num_samples += features.size(0)
    iteration_time = time.time() - start_time

    print(
        f"DataLoader iteration time: {iteration_time:.4f}s for {num_samples} samples ({num_batches} batches)"
    )
    print(f"Iteration speed: {num_samples/iteration_time:.2f} samples/s")
    print("Performance checks passed!\n")


if __name__ == "__main__":
    test_correctness()
    test_performance()

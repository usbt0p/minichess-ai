import random
import torch
import os

class RandomBaseline():
    """
    Random Baseline for 5x5 Minichess.
    Input: Flattened one-hot encoded board (5^2 * (12 + 1) = 325)
        5^2 squares, 12 piece types (6 white, 6 black), 1 extra input for empty squares
    Output: random predictions. for now, just pick one out of a uniform distribution (no top-k).

    - result_mode: if the model picks the game result as a value in [-1,1] (regression) or as a 
    class in (0,1,2) (classification). 
    
    """
    
    def __init__(self, batchsize=1, policy_size=600, result_mode= "classification"):
        self.batchsize = batchsize
        self.policy_size = policy_size
        self.result_mode = result_mode

    def _pick_random_move(self):
        """Pick a random move from the legal moves."""
        return torch.tensor(random.randint(0, self.policy_size-1), dtype=torch.long)

    def _pick_random_result(self):
        """Pick a random result from the possible results."""
        if self.result_mode == "classification":
            return torch.tensor(random.choice([0, 1, 2]), dtype=torch.long)
        else:
            return torch.tensor(random.uniform(-1, 1), dtype=torch.float)

    def forward(self):
        """Predict the next move and result from the current state."""
        # TODO legal_moves, and masking for illegal moves
        moves = []
        results = []
        for _ in range(self.batchsize):
            moves.append(self._pick_random_move())
            results.append(self._pick_random_result())
        return torch.stack(moves), torch.stack(results)

    def __call__(self):
        return self.forward()

if __name__ == "__main__":
    from src.models.dataloaders import get_dataloaders, MinichessTextDataset
    import os

    rb = RandomBaseline(batchsize=128, result_mode="classification")
    # print(rb()) # Quitamos el print inicial

    SOURCE_FILE = "data/gardner_depth2/gen_gardner_d2.txt"
    TARGET_DIR = "data/subsets_d2"
    OUTPUT_DIR = "src/benchmarks/random_baseline"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    SIZES = [50_000, 100_000, 500_000, 1_000_000, 1_800_000, 3_000_000, 6_000_000, 10_000_000]

    for size in SIZES:
        data_path = os.path.join(TARGET_DIR, f"subset_{size}.txt")
        if not os.path.exists(data_path):
            print(f"Skipping {size}, file not found: {data_path}")
            continue
            
        print(f"Benchmarking Random Baseline for size {size}...")

        # load dataset
        dataset = MinichessTextDataset(data_path, use_cache=True, time=False)

        # get dataloaders
        train_loader, val_loader = get_dataloaders(
            dataset, batch_size=256, train_ratio=0.98, num_workers=8, time=False)

        # "predict" and calculate accuracies
        correct_moves = 0
        correct_results = 0
        total = 0
        
        # We only need to iterate the validation set to get the validation accuracy
        for boards, moves, results, scores in val_loader:
            batch_size = boards.size(0)
            total += batch_size
            
            # Re-initialize random baseline for the correct batch size
            rb = RandomBaseline(batchsize=batch_size, result_mode="classification")
            pred_moves, pred_results = rb()
            
            # Calculate correct predictions
            correct_moves += (pred_moves == moves).sum().item()
            correct_results += (pred_results == results.squeeze()).sum().item()
            
        move_acc = 100 * correct_moves / total if total > 0 else 0
        res_acc = 100 * correct_results / total if total > 0 else 0
        
        print(f"  Move Acc: {move_acc:.2f}% | Result Acc: {res_acc:.2f}%")
        
        # Write to log file
        log_path = os.path.join(OUTPUT_DIR, f"logs_{size}.txt")
        with open(log_path, "w") as f:
            f.write(f"Best move accuracy: {move_acc:.2f}%\n")
            f.write(f"Best result accuracy: {res_acc:.2f}%\n")
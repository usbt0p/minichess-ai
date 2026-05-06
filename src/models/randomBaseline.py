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
    
    def __init__(self, batchsize=1, policy_size=704, result_mode="classification"):
        self.batchsize = batchsize
        self.policy_size = policy_size
        self.result_mode = result_mode

    def _pick_random_move(self):
        """Pick a random move from all possible moves."""
        return torch.tensor(random.randint(0, self.policy_size-1), dtype=torch.long)

    def _pick_random_result(self):
        """Pick a random result from the possible results."""
        if self.result_mode == "classification":
            return torch.tensor(random.choice([0, 1, 2]), dtype=torch.long)
        else:
            return torch.tensor(random.uniform(-1, 1), dtype=torch.float)

    def forward(self, masks=None):
        """Predict the next move and result from the current state."""
        results = []
        for _ in range(self.batchsize):
            results.append(self._pick_random_result())
            
        moves = []
        if masks is not None:
            # For each item in the batch, pick a random valid move
            for i in range(self.batchsize):
                valid_indices = torch.nonzero(masks[i]).squeeze()
                if valid_indices.dim() == 0:  # Only one valid move
                    moves.append(torch.tensor(valid_indices.item(), dtype=torch.long))
                elif len(valid_indices) > 0:  # Multiple valid moves
                    idx = random.choice(valid_indices.tolist())
                    moves.append(torch.tensor(idx, dtype=torch.long))
                else:  # Fallback if no legal moves (shouldn't happen, so better raise)
                    raise ValueError("No legal moves found for the current state")
                    #moves.append(self._pick_random_move())
        else:
            # Pure random move across the entire policy head
            for _ in range(self.batchsize):
                moves.append(self._pick_random_move())
                
        return torch.stack(moves), torch.stack(results)

    def __call__(self, masks=None):
        return self.forward(masks=masks)

if __name__ == "__main__":
    from src.models.dataloaders import get_dataloaders, MinichessTextDataset
    import os

    # Usamos los nuevos datasets con coronaciones
    #SOURCE_FILE = "data/gardner_depth2/d2_with_promotions.txt"
    TARGET_DIR = "data/subsets_d2_promotions"
    OUTPUT_DIR = "src/benchmarks/random_mask_prom"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    SIZES = [50_000, 100_000, 500_000, 1_000_000, 1_800_000, 3_000_000, 6_000_000, 10_000_000]

    for size in SIZES:
        data_path = os.path.join(TARGET_DIR, f"subset_{size}.txt")
        if not os.path.exists(data_path):
            print(f"Skipping {size}, file not found: {data_path}")
            continue
            
        print(f"Benchmarking Random Baseline for size {size}...")

        # load dataset asegurando pasar promotions=True
        dataset = MinichessTextDataset(data_path, promotions=True, use_cache=True, time=False)

        # get dataloaders
        train_loader, val_loader = get_dataloaders(
            dataset, batch_size=256, train_ratio=0.98, num_workers=8, time=False)

        # "predict" and calculate accuracies
        correct_moves_nomask = 0
        correct_moves_mask = 0
        correct_results = 0
        total = 0
        
        # We iterate the validation set to get 5 elements now
        for boards, moves, results, scores, masks in val_loader:
            batch_size = boards.size(0)
            total += batch_size
            
            # Re-initialize random baseline for the correct batch size
            rb = RandomBaseline(policy_size=704, batchsize=batch_size, result_mode="classification")
            
            # Pura aleatoriedad (sin máscaras)
            # pred_moves_nomask, pred_results = rb(masks=None)
            # correct_moves_nomask += (pred_moves_nomask == moves).sum().item()
            
            # Aleatoriedad legal (con máscaras)
            pred_moves_mask, pred_results = rb(masks=masks)
            
            # Calculate correct predictions
            correct_results += (pred_results == results.squeeze()).sum().item()
            correct_moves_mask += (pred_moves_mask == moves).sum().item()
            
        #move_acc_nomask = 100 * correct_moves_nomask / total if total > 0 else 0
        move_acc_mask = 100 * correct_moves_mask / total if total > 0 else 0
        res_acc = 100 * correct_results / total if total > 0 else 0
        
        # print(f"  Move Acc (No Mask): {move_acc_nomask:.2f}%")
        print(f"  Move Acc (Masked): {move_acc_mask:.2f}%")
        print(f"  Result Acc: {res_acc:.2f}%")
        
        # Write to log file
        log_path = os.path.join(OUTPUT_DIR, f"logs_{size}.txt")
        with open(log_path, "w") as f:
            #f.write(f"Best move accuracy: {move_acc_nomask:.2f}%\n")
            f.write(f"Best masked move accuracy: {move_acc_mask:.2f}%\n")
            f.write(f"Best result accuracy: {res_acc:.2f}%\n")
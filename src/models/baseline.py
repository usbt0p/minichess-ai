import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import time
from src.utils.utils import time_this

from src.models.dataloaders import get_dataloaders, MinichessTextDataset


# TODO dataloader from text and dataloader from binary
# TODO (maybe) pre-compute the legal move / illegal move mask to apply in the forward pass

class BaselineNet(nn.Module):
    """
    MLP Baseline for 5x5 Minichess.
    Input: Flattened one-hot encoded board (5^2 * (12 + 1) = 325)
        5^2 squares, 12 piece types (6 white, 6 black), 1 extra input for empty squares
    Output: Policy logits (max possible moves) and Value (-1 to 1)
        policy_size = 5^2 * 5^2 = 625 (number of possible moves from any square to any square), but if we take 
        only the moves to other squares, we have 5^2 * (5^2 - 1) = 600
    No moves for en passant or castling, because they are not allowed in 5x5 Minichess.
    
    TODO promotion still has no encoding.
    """
    def __init__(self, input_size=325, hidden_size=512, policy_size=600):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size*2)
        self.fc3 = nn.Linear(hidden_size*2, hidden_size)

        # value head predicts both the game result (-1/0/1) -> mapped to (0,1,2) for classes
        # and the score of the evaluation function, positive or negative.
        # for this we make two separate heads:
        
        # NOTE this might not be te bes, since we are making a very "similar" signal to the target, but with different scale.
        # a better idea might be to have the result be predicted as categorical with 3 classes
    
        self.value_result_head = nn.Linear(hidden_size, 3)
        # and we combine the losses in the training loop.

        # policy head predicts the probability of each move (600 possible moves)
        self.policy_head = nn.Linear(hidden_size, policy_size)

    def forward(self, x):
        # x shape: (Batch, 325)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        value_result = self.value_result_head(x) # logits without tanh because we predict 0/1/2
        # value_score = self.value_score_head(x) # no activation because it's an unbounded regression  
        # value = value_result + value_score # combine the two heads

        policy_logits = self.policy_head(x) 
        # TODO apply mask to policy_logits to remove illegal moves

        return policy_logits, value_result

@time_this
def train_model(
    train_loader, 
    val_loader,
    num_epochs=10,
    patience=5, 
    lr=2e-3,
    weight_decay=2e-5,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    '''
    - Patience: number of epochs worsened validation loss until early stopping
    '''

    model = BaselineNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    print(f"Using device: {device}")

    policy_criterion = nn.CrossEntropyLoss()
    value_result_criterion = nn.CrossEntropyLoss()
    # value_score_criterion = nn.MSELoss()

    # TODO wandb logging after everything else is done

    train_losses = [] # list of train loss per epoch, for all the three losses
    val_losses = []
    prev_validation_loss = float("-inf")

    val_move_accs = []
    val_res_accs = []
    
    patience_count = 0
    debug_flag = True
    
    best_move_acc = float("-inf")
    best_result_acc = float("-inf")

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_policy_loss, total_value_loss = 0.0, 0.0, 0.0

        start_time = time.time()

        for features, moves, results, scores in train_loader:
            features, moves, results, scores = features.to(device), moves.to(device), results.to(device), scores.to(device)

            optimizer.zero_grad()

            policy_logits, value_result = model(features)

            if debug_flag: # useful for debugging tensor dims
                print("results: ", results.shape)
                print("value_result: ", value_result.shape)
                print("moves: ", moves.shape)
                print("policy_logits: ", policy_logits.shape)
                print("\n")
                debug_flag = False

            policy_loss = policy_criterion(policy_logits, moves)
            value_result_loss = value_result_criterion(value_result, results)
            #value_score_loss = value_score_criterion(value_score, scores)
            # value_loss = value_result_loss + value_score_loss

            loss = policy_loss + value_result_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_result_loss.item()

        total_loss /= len(train_loader)
        total_policy_loss /= len(train_loader)
        total_value_loss /= len(train_loader)
        train_losses.append((total_loss, total_policy_loss, total_value_loss))

        epoch_time = time.time() - start_time

        model.eval()
        val_loss, correct_moves, correct_results, total_val_samples = 0.0, 0, 0, 0

        with torch.no_grad():
            for features, moves, results, scores in val_loader:
                features, moves, results, scores = features.to(device), moves.to(device), results.to(device), scores.to(device)

                policy_logits, value_result = model(features)

                policy_loss = policy_criterion(policy_logits, moves)
                value_result_loss = value_result_criterion(value_result, results)
                #value_score_loss = value_score_criterion(value_score, scores)
                # value_loss = value_result_loss + value_score_loss

                val_loss += (policy_loss + value_result_loss).item()

                _, predicted_moves = torch.max(policy_logits, 1)
                correct_moves += (predicted_moves == moves).sum().item()
                total_val_samples += moves.size(0)

                _, predicted_results = torch.max(value_result, 1)
                correct_results += (predicted_results == results).sum().item()
                

        val_move_acc = correct_moves / total_val_samples if total_val_samples > 0 else 0 
        val_res_acc = correct_results / total_val_samples if total_val_samples > 0 else 0 
        
        val_loss /= len(val_loader)
        # val_acc is already correct_moves / total_val_samples, do not divide by len(val_loader)
        val_losses.append(val_loss)
        val_move_accs.append(val_move_acc)
        val_res_accs.append(val_res_acc)

        print(f"Epoch {epoch+1}/{num_epochs} [{epoch_time:.2f}s]")
        print(f"  Train Loss: {total_loss:.4f} (Policy: {total_policy_loss:.4f}, Value: {total_value_loss:.4f})")
        print(f"  Val Loss:   {val_loss:.4f} | Val Move Acc: {val_move_acc*100:.2f}% | Val Result Acc: {val_res_acc*100:.2f}%")
        
        # consider the best model as the one with the best mean acc
        if (val_move_acc + val_res_acc)/2 > (best_move_acc + best_result_acc)/2:
            best_move_acc = val_move_acc
            best_result_acc = val_res_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "best_model.pth")
            
        # early stopping based on validation loss
        if patience > 0:
            if val_loss > prev_validation_loss:
                prev_validation_loss = val_loss
                patience_count += 1
                if patience_count == patience:
                    break
            else:
                patience_count = 0
                prev_validation_loss = val_loss

    print(f"Best mean accuracy: {(best_move_acc + best_result_acc)/2*100:.2f}% achieved at epoch {best_epoch}")
    print(f"Best move accuracy: {best_move_acc*100:.2f}%")
    print(f"Best result accuracy: {best_result_acc*100:.2f}%")

    return train_losses, val_losses, val_move_accs, val_res_accs, model


def plot_loss(train_losses, val_losses, val_move_accs, val_res_accs):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(train_losses)), [l[0] for l in train_losses], label='Train Loss')
    plt.plot(range(len(val_losses)), val_losses, label='Val Loss')
    
    # overlap policy and value loss with dashed lines
    plt.plot(range(len(train_losses)), [l[1] for l in train_losses], label='Train Policy Loss', linestyle='--')
    plt.plot(range(len(train_losses)), [l[2] for l in train_losses], label='Train Value Loss', linestyle='--')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('train_loss.png')
    plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(val_move_accs)), val_move_accs, label='Val Move Acc')
    plt.plot(range(len(val_res_accs)), val_res_accs, label='Val Result Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.savefig('val_accuracy.png')
    plt.show()

def validation_test(model, val_loader, device="cuda"):
    model = model.to(device)
    model.eval()
    
    correct_moves = 0
    correct_results = 0
    total_val_samples = 0

    with torch.no_grad():
        for features, moves, results, scores in val_loader:

            features, moves, results, scores = features.to(device), moves.to(device), results.to(device), scores.to(device)
            policy_logits, value_result = model(features)
            _, predicted_moves = torch.max(policy_logits, 1)
            correct_moves += (predicted_moves == moves).sum().item()
            total_val_samples += moves.size(0)

            _, predicted_results = torch.max(value_result, 1)
            correct_results += (predicted_results == results).sum().item()
    print("\n\nValidation test results:\n")
    print("\tTotal samples: ", total_val_samples)
    print("\tMove Accuracy: ", correct_moves / total_val_samples)
    print("\tResult Accuracy: ", correct_results / total_val_samples)


if __name__ == '__main__':
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/training_data_sample.txt"

    # load dataset
    dataset = MinichessTextDataset(data_path, time=True)

    # get dataloaders
    train_loader, val_loader = get_dataloaders(dataset, batch_size=256, num_workers=8, time=True)

    train_losses, val_losses, val_move_accs, val_res_accs, model = train_model(
        train_loader, val_loader, num_epochs=20, patience=4, time=True
    )
    plot_loss(train_losses, val_losses, val_move_accs, val_res_accs)

    validation_test(model, val_loader, device="cuda")

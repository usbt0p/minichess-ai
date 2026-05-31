import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import time
from src.utils.utils import time_this, count_params

from src.models.dataloaders import get_dataloaders, MinichessFfnDataset


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
    
    def __init__(self, input_size=325, hidden_size=512, policy_size=704, result_mode= "classification"):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size*2)
        self.fc3 = nn.Linear(hidden_size*2, hidden_size)

        dropout_prob = 0.1
        self.dropout = nn.Dropout(dropout_prob)
        
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size*2)
        self.bn3 = nn.BatchNorm1d(hidden_size)

        self.result_mode = result_mode
        if result_mode == "classification":
            self.out_result_size = 3
        elif result_mode == "regression":
            self.out_result_size = 1
        else:
            raise ValueError(f"Invalid result_mode: {result_mode}")

        self.value_result_head = nn.Linear(hidden_size, self.out_result_size)

        # policy head predicts the probability of each base move (600 possible moves + 104 promotions)
        self.policy_head = nn.Linear(hidden_size, policy_size)

    def forward(self, x, mask=None):
        # x shape: (Batch, 325)
        x = self.bn1(self.fc1(x))
        x = F.relu(x)
        x = self.bn2(self.fc2(x))
        x = F.relu(x)
        x = self.bn3(self.fc3(x))
        x = F.relu(x)
        x = self.dropout(x) # ensure dropout after all bn to prevent instabilities

        value_result = self.value_result_head(x) # logits without tanh if we predict 0/1/2
        if self.result_mode == "regression":
            # apply tanh to map values to (-1, 1) if doing regression
            value_result = torch.tanh(value_result)

        policy_logits = self.policy_head(x) 

        if mask is not None:
            policy_logits = policy_logits.masked_fill(~mask, -1e9)

        return policy_logits, value_result

@time_this
def train_model(
    model,
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

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    print(f"Using device: {device}")

    policy_criterion = nn.CrossEntropyLoss()
    if model.result_mode == "classification":
        value_result_criterion = nn.CrossEntropyLoss()
    else:
        value_result_criterion = nn.MSELoss()

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

        for features, moves, results, scores, masks in train_loader:
            features, moves, results, scores, masks = features.to(device), moves.to(device), results.to(device), scores.to(device), masks.to(device)

            optimizer.zero_grad()

            policy_logits, value_result = model(features, masks)

            if debug_flag: # useful for debugging tensor dims
                print("results: ", results.shape)
                print("value_result: ", value_result.shape)
                print("moves: ", moves.shape)
                print("policy_logits: ", policy_logits.shape)
                print("\n")
                debug_flag = False

            policy_loss = policy_criterion(policy_logits, moves)
            if model.result_mode == "classification":
                value_result_loss = value_result_criterion(value_result, results.long())
            else:
                value_result_loss = value_result_criterion(value_result.squeeze(-1), results.float())
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
            for features, moves, results, scores, masks in val_loader:
                features, moves, results, scores, masks = features.to(device), moves.to(device), results.to(device), scores.to(device), masks.to(device)

                policy_logits, value_result = model(features, masks)

                policy_loss = policy_criterion(policy_logits, moves)
                
                if model.result_mode == "classification":
                    value_result_loss = value_result_criterion(value_result, results.long())
                    _, predicted_results = torch.max(value_result, 1)
                    correct_results += (predicted_results == results).sum().item()
                else:
                    value_result_loss = value_result_criterion(value_result.squeeze(-1), results.float())
                    # For regression, mapping [-1, 1] back to class integers to check correctness
                    predicted_results = torch.round(value_result.squeeze(-1))
                    correct_results += (predicted_results == results).sum().item()

                val_loss += (policy_loss + value_result_loss).item()

                _, predicted_moves = torch.max(policy_logits, 1)
                correct_moves += (predicted_moves == moves).sum().item()
                total_val_samples += moves.size(0)

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
        for features, moves, results, scores, masks in val_loader:

            features, moves, results, scores, masks = features.to(device), moves.to(device), results.to(device), scores.to(device), masks.to(device)
            policy_logits, value_result = model(features, masks)
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
    
    model = BaselineNet(result_mode="classification").to("cuda")
    count_params(model)

    # load dataset
    dataset = MinichessFfnDataset(data_path, promotions=True, use_cache=True, time=True)

    # get dataloaders
    train_loader, val_loader = get_dataloaders(
        dataset, batch_size=256, train_ratio=0.98, num_workers=12, time=True)


    train_losses, val_losses, val_move_accs, val_res_accs, model = train_model(
        model, train_loader, val_loader, num_epochs=10, patience=4, time=True
    )
    #plot_loss(train_losses, val_losses, val_move_accs, val_res_accs)
    # use best model for validation test
    model.load_state_dict(torch.load("best_model.pth"))
    validation_test(model, val_loader, device="cuda")

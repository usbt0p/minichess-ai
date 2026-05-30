import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import time
from src.utils.utils import time_this, count_params

from src.models.dataloaders import get_dataloaders, MinichessTextDataset

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

    train_losses = [] # list of train loss per epoch, for all the three losses
    val_losses = []
    prev_validation_loss = float("-inf")

    val_move_accs = []
    val_res_accs = []
    
    patience_count = 0
    debug_flag = True
    
    best_move_acc = float("-inf")
    best_result_acc = float("-inf")

    # TODO criterion

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_policy_loss, total_value_loss = 0.0, 0.0, 0.0

        start_time = time.time()

        # TODO training loop
        ...    

    print(f"Best mean accuracy: {(best_move_acc + best_result_acc)/2*100:.2f}% achieved at epoch {best_epoch}")
    print(f"Best move accuracy: {best_move_acc*100:.2f}%")
    print(f"Best result accuracy: {best_result_acc*100:.2f}%")


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
    '''
    Tests the model on the validation set and returns the accuracy for moves and results.
    '''
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

    # # load dataset
    # dataset = MinichessTextDataset(data_path, promotions=True, use_cache=True, time=True)

    # # get dataloaders
    # train_loader, val_loader = get_dataloaders(
    #     dataset, batch_size=256, train_ratio=0.98, num_workers=12, time=True)

    # train_losses, val_losses, val_move_accs, val_res_accs, model = train_model(
    #     model, train_loader, val_loader, num_epochs=10, patience=4, time=True
    # )
    # #plot_loss(train_losses, val_losses, val_move_accs, val_res_accs)
    # # use best model for validation test
    # model.load_state_dict(torch.load("best_model.pth"))
    # validation_test(model, val_loader, device="cuda")
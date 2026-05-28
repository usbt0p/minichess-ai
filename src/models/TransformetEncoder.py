import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import time

# pyrefly: ignore [missing-import]
from src.utils.utils import time_this, count_params

# pyrefly: ignore [missing-import]
from src.models.dataloaders import get_dataloaders, MinichessTextDataset

class MLP(nn.Module):
    '''
    Multi-layer perceptron for transformer block.
    
    Architecture:
        1. linear expand
        2. gelu
        4. linear reduce
        5. dropout
    '''

    def __init__(self, embedding_dim, dropout, expand_factor=3):
        super(MLP, self).__init__()
        
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, expand_factor * embedding_dim),
            nn.GELU(),
            # optionally, other dropout here
            nn.Linear(expand_factor * embedding_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ffn(x)
    

class TransformerBlock(nn.Module):
    '''
    Input: (seq_len, batch, embedding_dim) = (27[vector], b, d_k). 27 because the size of the input vector. 
    Output: (seq_len, batch, embedding_dim) = (27, b, d_k). output in transformer blocks usually 
        doesn't change the size of the input vector. we're just "transforming"!
    
    Architecture:
    0. residual stream from previous layer (or input to the first layer) 
    1. pre-rmsnorm 
    2. multihead attention with H heads
    3. add residual
    4. pre-rmsnorm
    5. mlp
    6. add residual
    '''

    def __init__(self, embedding_dim, num_heads, mha_dropout, mlp_dropout):
        super(TransformerBlock, self).__init__()
        
        self.norm1 = torch.nn.RMSNorm(embedding_dim)
        # this is gold for understanding internals of pytorch MHA + flashattn (which torch auto uses if available (torch >= 2.0.0))
        # https://dev-discuss.pytorch.org/t/understanding-multi-head-attention-for-ml-framework-developers/1792
        self.mha = nn.MultiheadAttention(embedding_dim, num_heads, dropout=mha_dropout)
        self.norm2 = torch.nn.RMSNorm(embedding_dim)
        self.mlp = MLP(embedding_dim, mlp_dropout)

    def forward(self, x):
        residual = x 
        norm = self.norm1(x)
        attn_out = self.mha(query=norm, key=norm, value=norm, mask=None, 
                                    need_weights=False) 
                                    # Set need_weights=False to use the optimized scaled_dot_product_attention 
                                    # and achieve the best performance for MHA. 
                                    # https://docs.pytorch.org/docs/2.12/generated/torch.nn.MultiheadAttention.html
        # the last dropout in attn with need_weights=False is made after the attn weights, but before projection with W_o

        # https://github.com/pytorch/pytorch/blob/4f4b931aba66ae438aae8daca1dcbebeabb947e4/torch/nn/functional.py#L5504
        # so we could add another dropout after it here like this https://github.com/karpathy/nanoGPT/blob/master/model.py#L75
        # probably not so deep though
        x = residual + attn_out
        residual = x
        x = residual + self.mlp(self.norm2(x))
        return x

class MiniChessTransformerEncoder(nn.Module):
    """
    Transformer encoder for 5x5 Minichess.
    
    Input: flat vector of 27 bytes: 25 for the board, with a unique int for each piece type, 
        1 for the number of repetitions and 1 for the number of moves until the 50-move rule kicks in.
    Output: 704 neurons for policy (600 for moves (25*24) + 104 (13*4*2) for promotions), 1 for result. 
        No moves for en passant or castling, because they are not allowed in 5x5 Minichess.

    Architecture:
        1. Input layer: flat vector of 27 bytes.
        2. Embedding layer: embedding size of d_k = 256. map categorical vector data to a continuous vector space
        3. Positional encoding: simple sinusoidal positional encoding, but 2D. 
        4. N transformer blocks:
            0. residual stream from previous layer (or input to the first layer) 
            1. pre-rmsnorm 
            2. multihead attention with H heads
            3. add residual
            4. pre-rmsnorm
            5. fnn
                1. linear expand
                2. gelu
                3. dropout
                4. linear reduce
                5. dropout
            6. add residual
        5. final norm
        6. dual heads:
            1. policy head
                1. linear
                2. dropout
                3. linear
            2. value head
                1. linear
                2. dropout
                3. linear
    """
    
    def __init__(self, input_size, d_k, num_heads, num_blocks, mlp_dropout, mha_dropout):
        super(MiniChessTransformerEncoder, self).__init__()

        # some hyperparams we need before initializing layers
        self.input_size = input_size
        self.d_k = d_k
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        self.mlp_dropout = 0.1
        self.mha_dropout = 0.1
        self.policy_size = 704
        self.value_size = 1

        self.embedding = nn.Embedding(input_size, d_k)
        self.positional_encoding = nn.Parameter(torch.randn(1, input_size, d_k))
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_k, num_heads, mlp_dropout, mha_dropout) for _ in range(num_blocks)])
        
        self.policy_head = nn.Sequential(
            nn.Linear(),
            nn.GELU(),
            nn.Dropout(self.mlp_dropout),
            nn.Linear(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(),
            nn.GELU(),
            nn.Dropout(self.mlp_dropout),
            nn.Linear(),
        )

    def forward(self, x, mask=None):
        

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
    
    model = MiniChessTransformerEncoder().to("cuda")
    count_params(model)

    # TODO first, dummy pass to check proper network architecture
    ...

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

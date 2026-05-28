from torch.nn.modules import loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

from dataclasses import dataclass
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

    def __init__(self, config : EncoderConfig):
        super(MLP, self).__init__()
        
        self.ffn = nn.Sequential(
            nn.Linear(config.embbed_dim, config.mlp_expand_factor * config.embed_dim),
            nn.GELU(),
            # optionally, other dropout here
            nn.Linear(config.mlp_expand_factor * config.embed_dim, config.embed_dim),
            nn.Dropout(config.mlp_dropout),
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

    def __init__(self, config : EncoderConfig):
        super(TransformerBlock, self).__init__()

        self.norm1 = torch.nn.RMSNorm(config.embed_dim)
        # this is gold for understanding internals of pytorch MHA + flashattn (which torch auto uses if available (torch >= 2.0.0))
        # https://dev-discuss.pytorch.org/t/understanding-multi-head-attention-for-ml-framework-developers/1792
        self.mha = nn.MultiheadAttention(config.embed_dim, config.num_heads, dropout=config.mha_dropout)
        self.norm2 = torch.nn.RMSNorm(config.embed_dim)
        self.mlp = MLP(config.embed_dim, config.mlp_dropout)

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

class ChessEmbeddingSimple(nn.Module):
    """Maps a 27-element flat chess state into a unified embedding space.
    - 0-24: each of the 5x5 board squares, with a piece id inside
    - 25: repetitions for the 3 move rule
    - 26: halfmove for the 50 move rule
    """

    def __init__(self, config):
        super().__init__()
        # Total vocabulary size determined by the offsets (0 to 67)
        self.embedding = nn.Embedding(config.vocab_size, config.embbed_dim)

    def forward(self, board_flat, repetitions, halfmove_50):
        """
        Expects:
            board_flat: Tensor of shape (B, 25) with values [0, 12]
            repetitions: Tensor of shape (B, 1) with values [0, 2] (or dummy values)
            halfmove_50: Tensor of shape (B, 1) with values [0, 50]
        """
        # Apply structural offsets to prevent token ID collisions
        rep_shifted = repetitions + 13
        halfmove_shifted = halfmove_50 + 16

        # Combine into a single sequence of 27 tokens
        flat_state = torch.cat([board_flat, rep_shifted, halfmove_shifted], dim=1)
        
        # Output shape: (B, 27, embedding_dim)
        return self.embedding(flat_state)

@dataclass
class EncoderConfig:
    embbed_dim : int
    num_heads : int
    num_blocks : int

    input_size : int = 27
    vocab_size : int = 68
    mlp_dropout : float = 0.1
    mha_dropout : float = 0.1
    mlp_expand_factor : int = 4
    policy_size : int = 704
    value_size : int = 1


class MiniChessTransformerEncoder(nn.Module):
    """
    Transformer encoder for 5x5 Minichess.
    
    Input: flat vector of 27 bytes: 25 for the board, with a unique int for each piece type, 
        1 for the number of repetitions and 1 for the number of moves until the 50-move rule kicks in.
    Output: 704 neurons for policy (600 for moves (25*24) + 104 (13*4*2) for promotions), 1 for result. 
        No moves for en passant or castling, because they are not allowed in 5x5 Minichess.

    Architecture:
        1. Input layer: flat vector of 27 bytes.
        2. Embedding layer: embedding size of embbed_dim = 256. map categorical vector data to a continuous vector space
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

    def __init__(self, config : EncoderConfig):
        super(MiniChessTransformerEncoder, self).__init__()

        self._config = config

        self.backbone = nn.ModuleDict(
            dict(
                w_embed=ChessEmbeddingSimple(config),
                w_pos_embed=nn.Embedding(config.input_size, config.embbed_dim),
                dropout=nn.Dropout(config.dropout),
                transformer_blocks=nn.ModuleList(
                    [TransformerBlock(config) for _ in range(config.num_blocks)]
                ),
                # add a final norm since last transformer's ends with residual after mlp
                final_norm=nn.RMSNorm(config.embbed_dim),
            )
        )

        # TODO decide head linear dims, look at papers
        self.heads = nn.ModuleDict(
            dict(
                policy = nn.Sequential(
                    nn.Linear(config.embbed_dim, config.mlp_expand_factor * config.embbed_dim),
                    nn.GELU(),
                    nn.Dropout(config.mlp_dropout),
                    nn.Linear(config.mlp_expand_factor * config.embbed_dim, config.policy_size),
                ),
                value = nn.Sequential(
                    nn.Linear(config.embbed_dim, config.mlp_expand_factor * config.embbed_dim),
                    nn.GELU(),
                    nn.Dropout(config.mlp_dropout),
                    nn.Linear(config.mlp_expand_factor * config.embbed_dim, config.value_size),
                )
            )
        )

    def forward(self, input, targets=None):
        B, S = input.size() # TODO verify it's like this and not the opposite
        assert S == self._config.input_size # this should hold true
        device = input.device 

        # here we go, from karpathys nanoGPT

        # simple pos absolute pos embeddings
        pos = torch.arange(0, S, dtype=torch.long, device=device)

        tok_embed = self.backbone.w_embed(input)
        pos_embed = self.backbone.w_pos_embed(input)

        x = self.backbone.dropout(tok_embed + pos_embed)
        for block in self.backbone.transformer_blocks:
            x = block(x)
        x = self.transformer.final_norm(x)

        # # TODO see if i want to do something like this or do loss outside
        # if targets is not None:
        #     # if we are given some desired targets also calculate the loss
        #     logits = self.lm_head(x)
        #     loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        # else:
        #     # inference-time mini-optimization: only forward the lm_head on the very last position
        #     logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
        #     loss = None

        policy_logits = self.heads.policy(x)
        value_pred = self.heads.value(x)

        return policy_logits, value_pred


    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        Quasi-copied from nanoGPT
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.backbone.w_pos_embed.weight.numel()
            n_params -= self.backbone.w_embed.weight.numel()
        return n_params

    @classmethod
    def from_pretrained():
        # TODO once it's trained we'll need to load the 
        # weights back in for use in PPO 
        ...



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

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
            nn.Linear(config.embed_dim, config.mlp_expand_factor * config.embed_dim),
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
        self.mha = nn.MultiheadAttention(config.embed_dim, config.num_heads, dropout=config.mha_dropout, batch_first=True)
        self.norm2 = torch.nn.RMSNorm(config.embed_dim)
        self.mlp = MLP(config)

    def forward(self, x):
        residual = x 
        norm = self.norm1(x)
        attn_out, _ = self.mha(query=norm, key=norm, value=norm, attn_mask=None, need_weights=False) 
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
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)

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
    embed_dim : int
    num_heads : int
    num_blocks : int
    batch_size : int

    input_size : int = 27
    vocab_size : int = 68
    mlp_dropout : float = 0.1
    mha_dropout : float = 0.1
    embed_dropout : float = 0.1
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
        2. Embedding layer: embedding size of embed_dim = 256. map categorical vector data to a continuous vector space
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
                w_pos_embed=nn.Embedding(config.input_size, config.embed_dim),
                embed_dropout=nn.Dropout(config.embed_dropout),
                transformer_blocks=nn.ModuleList(
                    [TransformerBlock(config) for _ in range(config.num_blocks)]
                ),
                # add a final norm since last transformer's ends with residual after mlp
                final_norm=nn.RMSNorm(config.embed_dim),
            )
        )

        # TODO maybe modify arch for the heads, make them bigger?
        self.heads = nn.ModuleDict(
            dict(
                policy = nn.Sequential(
                    nn.Linear(config.embed_dim, config.mlp_expand_factor * config.embed_dim),
                    nn.GELU(),
                    nn.Dropout(config.mlp_dropout),
                    nn.Linear(config.mlp_expand_factor * config.embed_dim, config.policy_size),
                ),
                value = nn.Sequential(
                    nn.Linear(config.embed_dim, config.mlp_expand_factor * config.embed_dim),
                    nn.GELU(),
                    nn.Dropout(config.mlp_dropout),
                    nn.Linear(config.mlp_expand_factor * config.embed_dim, config.value_size),
                )
            )
        )

    def forward(self, input, targets=None):
        B, S = input.size() 
        assert (S == self._config.input_size) and (B == self._config.batch_size)
        device = input.device 

        # here we go, from karpathys nanoGPT

        # before forwarding, we must split the tensor and pass to the chess embedder
        board_flat, repetitions, halfmove_50 = torch.split(input, [25, 1, 1], dim=1)
        tok_embed = self.backbone.w_embed(board_flat, repetitions, halfmove_50)

        # simple pos absolute pos embeddings
        pos = torch.arange(0, S, dtype=torch.long, device=device) # shape (input_size, )
        pos_embed = self.backbone.w_pos_embed(pos) # gets converted to (input_size, emb_dim)

        x = self.backbone.embed_dropout(tok_embed + pos_embed) # sum is broadcasted across batch dimension
        for block in self.backbone.transformer_blocks:
            x = block(x)
        x = self.backbone.final_norm(x)

        # # TODO see if i want to do something like this or do loss outside
        # if targets is not None:
        #     # if we are given some desired targets also calculate the loss
        #     logits = self.lm_head(x)
        #     loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        # else:
        #     # inference-time mini-optimization: only forward the lm_head on the very last position
        #     logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
        #     loss = None
        print("shape before heads: ", x.shape)

        # TODO here's the problem! how do we go from (batch, seq_len, embed_d) to 
        # (B, policy) y (B, value) in the heads?
        # going from (B, 27, 256) to (B, 27 * 256) via x.view(B, -1) is valid and gets 
        # me a 6,912 input for the policy head. a linear layer going from 6,912 to 704 adds 
        # roughly 4.8 million parameters the policy head alone whichi is insane

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
            n_params -= self.backbone.w_embed.embedding.weight.numel()
        return n_params

    @classmethod
    def from_pretrained():
        # TODO once it's trained we'll need to load the 
        # weights back in for use in PPO 
        ...


if __name__ == '__main__':
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/training_data_sample.txt"

    config = EncoderConfig(
        embed_dim=256, 
        num_heads=8, 
        num_blocks=4, 
        batch_size=32
    )

    model = MiniChessTransformerEncoder(config).to("cuda")
    count_params(model)
    print("karpathys parameter count:", model.get_num_params(non_embedding=True))

    # TODO first, dummy pass to check proper network architecture
    # we want to go like this since it's the default for torch's mha
    dummy_board = torch.randint(0, 13, (config.batch_size, 25))
    dummy_rep = torch.randint(0, 3, (config.batch_size, 1))
    dummy_halfmove = torch.randint(0, 51, (config.batch_size, 1))
    dummy_tensor = torch.cat([dummy_board, dummy_rep, dummy_halfmove], dim=1).to("cuda")
    print("dummy tensor size: ", dummy_tensor.size())
    out = model(dummy_tensor)
    print("Policy logits shape:", out[0].shape)
    print("Value prediction shape:", out[1].shape)

    #returns:
    '''
    Total number of trainable parameters: 4430529
        In bits: 141776928 bits
        In bytes: 17722116 bytes
        In kilobytes: 17306.75390625 KB
        In megabytes: 16.901126861572266 MB

    karpathys parameter count: 4406209
    torch.Size([32, 27])
    shape before heads:  torch.Size([32, 27, 256])
    Policy logits shape: torch.Size([32, 704])
    Value prediction shape: torch.Size([32, 1])
    '''

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

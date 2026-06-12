import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from dataclasses import dataclass
import math

# pyrefly: ignore [missing-import]
from src.utils.utils import time_this, count_params


@dataclass
class EncoderConfig:
    '''MiniChess encoder configuration.
    Decouples config hyperparameters for easier logging, modification and tuning.    
    '''

    embed_dim: int
    num_heads: int
    num_blocks: int
    batch_size: int

    input_size: int = 27
    vocab_size: int = 68
    mlp_dropout: float = 0.1
    mha_dropout: float = 0.1
    embed_dropout: float = 0.1
    mlp_expand_factor: int = 4

    policy_head_hidden_dim : int = 64 # TODO important to tune this to not compress too much
    policy_size: int = 704
    promotion_size : int = 104
    value_size: int = 1

    custom_init: bool = False
    attn_backend: str = "auto"
    autocast_mode: str = "none"

    # for reference: https://www.pythonmorsels.com/customizing-dataclass-initialization/
    def __post_init__(self):
        assert self.policy_size == (25*24) + self.promotion_size, "Policy must be possible moves + possible promotions"
        # not mine! https://stackoverflow.com/questions/57025836/how-to-check-if-a-given-number-is-a-power-of-two
        # is_power_of_two = lambda n: (n & (n-1) == 0) and n != 0 
        # assert is_power_of_two(self.embed_dim) and \
        #     is_power_of_two(self.policy_head_hidden_dim), "Set these to powers of two for better efficiency"
        assert self.embed_dim % self.num_heads == 0, "Set the number of dims to be divisible by the number of heads."

class MLP(nn.Module):
    """
    Multi-layer perceptron for transformer block.

    Architecture:
        1. linear expand
        2. gelu
        4. linear reduce
        5. dropout
    """

    def __init__(self, config: EncoderConfig):
        super(MLP, self).__init__()

        self.ffn = nn.Sequential(
            nn.Linear(config.embed_dim, config.mlp_expand_factor * config.embed_dim),
            nn.GELU(),
            # optionally, other dropout here, OR THE MOVE THE NEXT HERE
            nn.Linear(config.mlp_expand_factor * config.embed_dim, config.embed_dim),
            nn.Dropout(config.mlp_dropout),
        )
        # Tag residual projection for special scaling in custom weight initialization
        self.ffn[2].residual_proj = True

    def forward(self, x):
        return self.ffn(x)


class TransformerBlock(nn.Module):
    """
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
    """

    def __init__(self, config: EncoderConfig):
        super(TransformerBlock, self).__init__()
        
        # Resolve attention backend during init to avoid conditional branching in forward()
        self.attn_backend = config.attn_backend
        if self.attn_backend == "auto":
            self.backend_enum = None
        elif self.attn_backend == "flash":
            self.backend_enum = SDPBackend.FLASH_ATTENTION
        elif self.attn_backend == "efficient":
            self.backend_enum = SDPBackend.EFFICIENT_ATTENTION
        elif self.attn_backend == "math":
            self.backend_enum = SDPBackend.MATH
        else:
            raise ValueError(f"Unknown attention backend: {self.attn_backend}")

        self.norm1 = torch.nn.RMSNorm(config.embed_dim)
        # this is gold for understanding internals of pytorch MHA + flashattn (which torch auto uses if available (torch >= 2.0.0))
        # https://dev-discuss.pytorch.org/t/understanding-multi-head-attention-for-ml-framework-developers/1792
        self.mha = nn.MultiheadAttention(
            config.embed_dim,
            config.num_heads,
            dropout=config.mha_dropout,
            batch_first=True, # simplify things and avoid transpositions
        )
        self.norm2 = torch.nn.RMSNorm(config.embed_dim)
        self.mlp = MLP(config)
        # Tag the output projection layer for special scaling in custom weight initialization
        self.mha.out_proj.residual_proj = True

    def forward(self, x):
        residual = x
        norm = self.norm1(x)

        if self.backend_enum is None:
            # torch will choose the best for us based on our dtype
            attn_out, _ = self.mha(
                query=norm, key=norm, value=norm, attn_mask=None, need_weights=False
                )
        else:
            # Set need_weights=False to use the optimized scaled_dot_product_attention and therefore flash attn
            # https://docs.pytorch.org/docs/2.12/generated/torch.nn.MultiheadAttention.html
            with sdpa_kernel(self.backend_enum):
                attn_out, _ = self.mha(
                    query=norm, key=norm, value=norm, attn_mask=None, need_weights=False
                    )
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

    def __init__(self, config: EncoderConfig):
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
        # TODO maybe it would be better to make this fail loudly...
        # Clamp repetitions and halfmove count to avoid out-of-bounds embedding index errors
        rep_clipped = torch.clamp(repetitions, 0, 2)
        halfmove_clipped = torch.clamp(halfmove_50, 0, 51)

        # Apply structural offsets to prevent token ID collisions
        rep_shifted = rep_clipped + 13
        halfmove_shifted = halfmove_clipped + 16

        # Combine into a single sequence of 27 tokens
        flat_state = torch.cat([board_flat, rep_shifted, halfmove_shifted], dim=1)

        # Output shape: (B, 27, embedding_dim)
        return self.embedding(flat_state)


class MatrixPolicyHead(nn.Module):
    """Computes square-to-square move transitions for the policy head
    The idea is pulled from Maia Chess, and aims to compress the backbone output
    from (batch, seq_len, embed_d) to (batch, policy) by exploiting the inductive bias of chess, 
    keeping relatively few params wrt performance and speed.
    For reference, see: https://github.com/CSSLab/maia3/blob/main/maia3/models.py#L371-L400
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.head_dim = config.policy_head_hidden_dim

        # these fully connected layers will act as conditioning for "from where" and "to were" we move
        self.proj_from = nn.Sequential(
            nn.Linear(config.embed_dim, config.policy_head_hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(config.policy_head_hidden_dim * 2, config.policy_head_hidden_dim),
        )
        self.proj_to = nn.Sequential(
            nn.Linear(config.embed_dim, config.policy_head_hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(config.policy_head_hidden_dim * 2, config.policy_head_hidden_dim),
        )

        # promotion predictions use context from the CLS token
        self.proj_promo = nn.Linear(config.embed_dim, config.promotion_size)

        # mask to remove from==to squares (25 entries), on the diagonal
        # the buffer is a tensor that should not be considered a model parameter
        eye_mask = torch.eye(25, dtype=torch.bool)
        self.register_buffer("valid_move_mask", ~eye_mask.view(-1))

    def forward(self, board_tokens, cls_token):
        B = board_tokens.size(0)
        assert board_tokens.size()[1] == 25, "25 board tokens must be passed"

        # project the tokens to a smaller dimension for multiplication
        sq_from = self.proj_from(board_tokens)  # (B, 25, head_dim)
        sq_to = self.proj_to(board_tokens)  # (B, 25, head_dim)

        # now we use the instein sum to compute an batched matrix multiplication (tensor outer product)
        # this gathers information about how "good" each (origin_square, destination_square) is
        # tensor output: (B, 25, 64) x (B, 64, 25) -> (B, 25, 25)
        scores = torch.einsum("bid,bjd->bij", sq_from, sq_to) / math.sqrt(self.head_dim)
        # now, we can flatten without getting an insane amount of params
        scores_flat = scores.reshape(B, 625) 

        # filter down to the 600 base valid coordinate transitions
        base_moves = scores_flat[:, self.valid_move_mask]  # (B, 600)

        # extract promotion probabilities separately, using global CLS vector
        promo_logits = self.proj_promo(cls_token)  # (B, 104)

        return torch.cat([base_moves, promo_logits], dim=1)  # (B, 704)


class MiniChessTransformerEncoder(nn.Module):
    """
    Transformer encoder for 5x5 Minichess.

    Input: flat vector of 27 bytes: 25 for the board, with a unique int for each piece type,
        1 for the number of repetitions and 1 for the number of moves until the 50-move rule kicks in.
    Output: 704 neurons for policy (600 for moves (25*24) + 104 (13*4*2) for promotions), 1 for result.
        No moves for en passant or castling, because they are not allowed in 5x5 Minichess.

    Architecture:
        1. Input layer: flat vector of 27 bytes.
        2. Embedding layer: embedding size of embed_dim = 256. 
            map categorical vector data to a continuous vector space and concat 1 CLS tokn
        3. Positional encoding: simple linear positional encoding. add 1 for CLS token
        4. N transformer blocks:
            0. residual stream from previous layer (or input to the first layer)
            1. pre-rmsnorm
            2. multihead attention with H heads
            3. add residual
            4. pre-rmsnorm
            5. fnn
                1. linear expand -> gelu -> dropout
                4. linear reduce -> dropout
            6. add residual
        5. final norm
        6. dual heads:
            1. policy head
                1. 2 linears for from_position and to_position
                2. outer product over batch dim of those, gives relation matrix
                3. linear for the CLS token (gives promotion info)
                4. concat both and return logits
            2. value head
                1. linear expand -> gelu -> dropout -> linear
    """

    def __init__(self, config: EncoderConfig):
        super(MiniChessTransformerEncoder, self).__init__()

        self._config = config

        # learnable global state token, NLP cross-encoder style. initialized to zeros for "no meaning"
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))


        self.backbone = nn.ModuleDict(
            dict(
                w_embed=ChessEmbeddingSimple(config),
                w_pos_embed=nn.Embedding(config.input_size + 1, config.embed_dim), # add CLS token
                embed_dropout=nn.Dropout(config.embed_dropout),
                transformer_blocks=nn.ModuleList(
                    [TransformerBlock(config) for _ in range(config.num_blocks)]
                ),
                # add a final norm since last transformer's ends with residual after mlp
                final_norm=nn.RMSNorm(config.embed_dim),
            )
        )

        self.heads = nn.ModuleDict(
            dict(
                policy=MatrixPolicyHead(config),
                # TODO decide if switch this to a simple linear
                value=nn.Sequential(
                    nn.Linear(
                        config.embed_dim, config.mlp_expand_factor * config.embed_dim
                    ),
                    nn.GELU(),
                    nn.Dropout(config.mlp_dropout), # TODO CAREFUL WITH THIS DURING PPO, SET TO model.eval() !!!!
                    nn.Linear(
                        config.mlp_expand_factor * config.embed_dim, config.value_size
                    ),
                    torch.nn.Tanh()
                ),
            )
        )

        # Apply custom weight initialization if enabled
        if config.custom_init:
            self.apply(self._init_weights)

        # Resolve autocast settings during initialization to avoid branch logic in forward()
        self.autocast_mode = getattr(config, 'autocast_mode', 'bfloat16')
        self.autocast_dtype = torch.bfloat16
        self.autocast_enabled_if_cuda = False

        # bfloat16 + flash attn + torch compile is fastest generally
        # bfloat sacrifices some performance due to lower fraction precision but is faster
        # numerical stability issues appear with float16 due to less exponent bits
        # float32 is the slowest and most memory demanding but gives bst metrics
        if self.autocast_mode == 'bfloat16':
            self.autocast_enabled_if_cuda = True
            self.autocast_dtype = torch.bfloat16
        elif self.autocast_mode == 'float16':
            self.autocast_enabled_if_cuda = True
            self.autocast_dtype = torch.float16
        elif self.autocast_mode == 'auto':
            self.autocast_enabled_if_cuda = True
            self.autocast_dtype = torch.bfloat16
        elif self.autocast_mode in ('float32', 'none', 'no', 'disabled'):
            self.autocast_enabled_if_cuda = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            # Check if this linear layer was tagged as a residual projection
            if getattr(module, 'residual_proj', False):
                std = std / math.sqrt(2 * self._config.num_blocks)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
            if hasattr(module, 'weight') and module.weight is not None:
                torch.nn.init.ones_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, input):
        '''Forward the MiniChess Encoder. This involves some moving parts for
        embedding, batched matmul and CLS token handling. 

        Args:
        - `input`: must be tensor of size `(batch_dim, seq_len=27, embed_dim)`. Will get transformed into
            `(batch_dim, seq_len=28, embed_dim)`.

        Out:
        - Policy logits tensor of size `(batch_dim, policy_size=704)`.
        - Value predictions tensor of size `(batch_dim, 1)`.
        '''
        B, S = input.size()
        assert S == self._config.input_size, f"Expected sequence length {self._config.input_size}, got {S}"
        device = input.device

        # before forwarding, we must split the tensor and pass to the chess embedder
        board_flat, repetitions, halfmove_50 = torch.split(input, [25, 1, 1], dim=1)
        
        # Check device type at runtime to configure autocast context manager
        autocast_enabled = (device.type == 'cuda' and self.autocast_enabled_if_cuda)

        with torch.autocast(device_type=device.type, dtype=self.autocast_dtype, enabled=autocast_enabled):
            tok_embed = self.backbone.w_embed(board_flat, repetitions, halfmove_50)

            # Prepend expanded CLS token array to sequence dimension
            cls_tokens = self.cls_token.expand(B, 1, -1)
            x = torch.cat([cls_tokens, tok_embed], dim=1)  # (B, 28, D)

            # simple pos absolute pos embeddings
            # TODO make these less shitty, maybe nn.RoPE or some learnable thing
            pos = torch.arange(0, S + 1, dtype=torch.long, device=device)  # shape (input_size, )
            pos_embed = self.backbone.w_pos_embed(pos)  # gets converted to (input_size, emb_dim)

            x = self.backbone.embed_dropout(x + pos_embed)  # sum is broadcasted across batch dimension
            for block in self.backbone.transformer_blocks:
                x = block(x)
            x = self.backbone.final_norm(x)
            
            # now: the problem! how do we go from (batch, seq_len, embed_d) to
            # (B, policy) y (B, value) in the heads? this is why we use CLS for value, and  the special policy head
            # with batch matmul between projections of the embedings

            # separate CLS from "board representation" (board tokens). 
            # metadata tokens are not explicitly used, but they have affected toks [0,26] thanks to the attention 
            cls_output = x[:, 0, :]           # (B, D)
            board_tokens = x[:, 1:26, :]      # (B, 25, D)

            policy_logits = self.heads.policy(board_tokens, cls_output)
            value_pred = self.heads.value(cls_output)

        # now, since we autocasted to bfloat16 previously, we need to go back to float32 to avoid 
        # numerical stability issues with the loss (like what a GradScaler does)
        return policy_logits.float(), value_pred.float() 

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


if __name__ == "__main__":

    # dummy pass to check proper network architecture
    config = EncoderConfig(embed_dim=256, num_heads=8, num_blocks=4, batch_size=32)

    model : MiniChessTransformerEncoder = MiniChessTransformerEncoder(config).to("cuda")
    count_params(model)
    print("karpathys parameter count:", model.get_num_params(non_embedding=True))

    # build a plausible tensor
    dummy_board = torch.randint(0, 13, (config.batch_size, 25))
    dummy_rep = torch.randint(0, 3, (config.batch_size, 1))
    dummy_halfmove = torch.randint(0, 51, (config.batch_size, 1))
    dummy_tensor = torch.cat([dummy_board, dummy_rep, dummy_halfmove], dim=1).to("cuda")
    print("dummy tensor size: ", dummy_tensor.size())
    
    # forward it and debug
    out = model.forward(dummy_tensor)
    print("Policy logits shape:", out[0].shape)
    print("Value prediction shape:", out[1].shape)

    # returns:
    """
    Total number of trainable parameters: 3505897
        In bits: 112188704 bits
        In bytes: 14023588 bytes
        In kilobytes: 13694.91015625 KB
        In megabytes: 13.37393569946289 MB

    karpathys parameter count (no embeddings): 3481321
    dummy tensor size:  torch.Size([32, 27])
    torch.Size([28, 256]) torch.Size([32, 28, 256])
    shape before heads:  torch.Size([32, 28, 256])
    Policy logits shape: torch.Size([32, 704])
    Value prediction shape: torch.Size([32, 1])
    """

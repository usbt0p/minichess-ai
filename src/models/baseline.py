import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        self.value_head = nn.Linear(hidden_size, 1)
        self.policy_head = nn.Linear(hidden_size, policy_size)

    def forward(self, x):
        # x shape: (Batch, 325)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        value = torch.tanh(self.value_head(x)) 
        policy_logits = self.policy_head(x) 
        # TODO apply mask to policy_logits to remove illegal moves
        
        return policy_logits, value

# TODO training loop

# TODO wandb logging after everyhing else is done

# TODO use loss as metric for now, but i need to find a wayt to evaluate: the real move acc / val acc?
# maybe play against a random oponnent
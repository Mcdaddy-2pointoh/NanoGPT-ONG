import torch 
from torch import nn
from torch.nn import functional as F
from model.layers.MaskedAttention import Head

class MultiHeadAttention(nn.Module):
    """
    Class: Implements multiple heads of self attention
    """

    def __init__(self, num_heads: int, block_size: int, n_embedd: int = 32, device: str = None, attention_head_size: int = 16, dropout: float = 0.2):
        """
        Function: Instances an object of class `MultiHeadAttention`
        Args:
            num_heads (int): Number of parallel heads to implement
            block_size (int): Block size is the maximum context window of the model
            n_embedd (int): Linear dimension in which the token in projected into
            device (str): The device on which the operation must be carried out `cuda` or `cpu`
            attention_head_size (int): The projection dimension of each individual attention head
            dropout (0 < float < 1): The value of dropout to be applied at layer
        """
        super().__init__()

        # Defining params of a single attention head
        self.attention_head_size = attention_head_size
        self.n_embedd = n_embedd
        self.block_size = block_size

        # Defining the number of attention heads
        self.num_heads = num_heads
    
        # Determine device type
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device in ["cuda", "cpu"]:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        device = self.device

        # Specifying the heads 
        self.heads = nn.ModuleList([Head(block_size=block_size, n_embedd=n_embedd, device=device, attention_head_size=attention_head_size, dropout=dropout)] * num_heads)

        # Projection of self attention to prepare for the residual skips
        self.proj = nn.Linear(n_embedd, n_embedd).to(device=device)

        # Dropout params and layer
        self.dropout = dropout
        self.Dropout = nn.Dropout(dropout).to(device=device)

    def forward(self, x):
        """
        Function: Feed forward function of the multiple attention heads 
        Args:
            x (torch.Tensor): Input params of the multiple attention heads 
        """
        # Get Multiheaded attention score
        out = torch.cat([head(x) for head in self.heads], dim=-1).to(device=self.device)

        # Project into the residual dimensions
        out = self.proj(out)

        # Include dropout
        out = self.Dropout(out)

        return out
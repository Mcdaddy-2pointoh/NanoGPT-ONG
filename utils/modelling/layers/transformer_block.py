import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.modelling.layers.attention_heads import MultiHeadAttention
from utils.modelling.layers.feed_forward import FeedForward

class Block(nn.Module):
    """
    Class: One single transformer block that can be replicated
    """

    def __init__(self, num_heads: int, block_size: int, n_embedd: int = 32, device: str = None, attention_head_size: int = 16):
        """
        Function: Instances an object of class `MultiHeadAttention`
        Args:
            num_heads (int): Number of parallel heads to implement
            block_size (int): Block size is the maximum context window of the model
            n_embedd (int): Linear dimension in which the token in projected into
            device (str): The device on which the operation must be carried out `cuda` or `cpu`
            attention_head_size (int): The projection dimension of all attention heads combined
        """
        super().__init__()

        # Attention params and methods
        individual_head_size = attention_head_size//num_heads
        self.individual_head_size = individual_head_size
        self.attention_head = MultiHeadAttention(num_heads = num_heads, n_embedd=n_embedd, block_size=block_size, device=device, attention_head_size=individual_head_size)

        # Feed forward params and methods
        self.ffn = FeedForward(attention_head_size=attention_head_size)

    def forward(self, x):
        """
        Function: Feed forward function of the model 
        Args:
            x (torch.Tensor): Input params of the model 
        """
        x = self.attention_head(x)
        x = self.ffn(x)

        return x
        
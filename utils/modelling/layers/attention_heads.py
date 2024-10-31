import torch 
from torch import nn
from torch.nn import functional as F

# Self attention head 
class Head(nn.Module):
    """
    Class:  One head of self attention
    """

    def __init__(self, block_size: int, n_embedd: int = 32, device: str = None, attention_head_size: int = 16):
        """
        Function: Instances an object of class `Head`
        Args:
            block_size (int): Block size is the maximum context window of the model
            n_embedd (int): Linear dimension in which the token in projected into
            device (str): The device on which the operation must be carried out `cuda` or `cpu`
            attention_head_size (int): The projection dimension of each individual attention head
        """

        super().__init__()
        self.attention_head_size = attention_head_size
        self.n_embedd = n_embedd
        self.block_size = block_size

        # Determine device type
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device in ["cuda", "cpu"]:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        device = self.device

        # Set attention parameters
        self.key = nn.Linear(n_embedd, attention_head_size, bias=False, device=device)
        self.value = nn.Linear(n_embedd, attention_head_size, bias=False, device=device)
        self.query = nn.Linear(n_embedd, attention_head_size, bias=False, device=device)
        self.register_buffer("tril", torch.tril(torch.ones((block_size, block_size))).to(device=device))

    def forward(self, x):
        """
        Function: Feed forward function of the multiple attention heads 
        Args:
            x (torch.Tensor): Input params of the multiple attention heads 
        """

        # Gather the shape of the input tensor
        B, T, C = x.shape

        # produce the key and query vectors for every token at time step in x
        k = self.key(x) # size (B, T, attention_head_size)
        kt = k.transpose(-2, -1).to(device=self.device) # size (B, attention_head_size, T)
        q = self.query(x) # size (B, T, attention_head_size)

        # Compute the attention scores
        wei = q @ kt * C **-0.5 # (B, T, attention_head_size) @ (B, attention_head_size, T) ------> (B, T, T)

        # Apply mask to ensure tokens cannot communicate with future tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        # Perform weighted aggregation
        out = wei @ self.value(x)

        return out

class MultiHeadAttention(nn.Module):
    """
    Class: Implements multiple heads of self attention
    """

    def __init__(self, num_heads: int, block_size: int, n_embedd: int = 32, device: str = None, attention_head_size: int = 16):
        """
        Function: Instances an object of class `MultiHeadAttention`
        Args:
            num_heads (int): Number of parallel heads to implement
            block_size (int): Block size is the maximum context window of the model
            n_embedd (int): Linear dimension in which the token in projected into
            device (str): The device on which the operation must be carried out `cuda` or `cpu`
            attention_head_size (int): The projection dimension of each individual attention head
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
        self.heads = nn.ModuleList([Head(block_size=block_size, n_embedd=n_embedd, device=device, attention_head_size=attention_head_size)] * num_heads)

    def forward(self, x):
        """
        Function: Feed forward function of the multiple attention heads 
        Args:
            x (torch.Tensor): Input params of the multiple attention heads 
        """
        return torch.cat([head(x) for head in self.heads], dim=-1)
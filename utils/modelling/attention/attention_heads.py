import torch 
from torch import nn
from torch.nn import functional as F

# Self attention head 
class Head(nn.Module):
    """
    Class:  One head of self attention
    """

    def __init__(self, block_size: int, n_embedd: int = 32, device: str = None, head_size: int = 16):
        super().__init__()
        self.head_size = head_size
        self.n_embedd = n_embedd

        # Determine device size
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device in ["cuda", "cpu"]:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        device = self.device

        # Set attention parameters
        self.key = nn.Linear(n_embedd, head_size, bias=False, device=device)
        self.value = nn.Linear(n_embedd, head_size, bias=False, device=device)
        self.query = nn.Linear(n_embedd, head_size, bias=False, device=device)
        self.register_buffer("tril", torch.tril(torch.ones((block_size, block_size))).to(device=device))

    def forward(self, x):

        # Gather the shape of the input tensor
        B, T, C = x.shape

        # produce the key and query vectors for every token at time step in x
        k = self.key(x) # size (B, T, head_size)
        kt = k.transpose(-2, -1).to(device=self.device) # size (B, head_size, T)
        q = self.query(x) # size (B, T, head_size)

        # Compute the attention scores
        wei = q @ kt * C **-0.5 # (B, T, head_size) @ (B, head_size, T) ------> (B, T, T)

        # Apply mask to ensure tokens cannot communicate with future tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        # Perform weighted aggregation
        out = wei @ self.value(x)

        return out

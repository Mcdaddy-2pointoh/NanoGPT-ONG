import torch 
from torch import nn
from torch.nn import functional as F
from positional_encoders.rotary_positional_encoding import RotaryPositionEncoding

# Self attention head 
class Head(nn.Module):
    """
    Class:  One head of self attention
    """

    def __init__(self, block_size: int, n_embedd: int = 32, device: str = None, attention_head_size: int = 16, dropout: float = 0.20, positional_encoder_type: str = None):
        """
        Function: Instances an object of class `Head`
        Args:
            block_size (int): Block size is the maximum context window of the model
            n_embedd (int): Linear dimension in which the token in projected into
            device (str): The device on which the operation must be carried out `cuda` or `cpu`
            attention_head_size (int): The projection dimension of each individual attention head
            dropout (0 < float < 1): The value of dropout to be applied at layer
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

        # Set the position Encoding Params
        if not isinstance(positional_encoder_type, str):
            raise TypeError("Argument `positional_encoder_type` must be of type str")
        
        # Set `positional_encoder_type` to naive
        elif positional_encoder_type == "naive":
            self.positional_encoder_type = "naive"

        # Set `positional_encoder_type` to sinusoidal
        elif positional_encoder_type == "sinusoidal":
            self.positional_encoder_type = "sinusoidal"

        # Set `positional_encoder_type` to RoPE
        elif positional_encoder_type == "RoPE":
            self.positional_encoder_type = "RoPE"

        # Else key `positional_encoder_type` is out of bounds raise error
        else:
            raise ValueError("Argument `positional_encoder_type` must be either 'RoPE', 'sinusoidal' or 'naive'")

        # Dropout params
        self.dropout = dropout
        self.Dropout = nn.Dropout(dropout).to(device=device)

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
        q = self.query(x) # size (B, T, attention_head_size)


        # If the positional_encoder_type is RoPE compute the rotated q, k
        if self.positional_encoder_type == "RoPE":
            q, k  = RotaryPositionEncoding(q, k, T, self.attention_head_size, device=self.device) # size (B, T, attention_head_size)

        kt = k.transpose(-2, -1).to(device=self.device) # size (B, attention_head_size, T)

        # Compute the attention scores
        wei = q @ kt * C **-0.5 # (B, T, attention_head_size) @ (B, attention_head_size, T) ------> (B, T, T)

        # Apply mask to ensure tokens cannot communicate with future tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        # Applying dropout 
        wei = self.Dropout(wei)

        # Perform weighted aggregation
        out = wei @ self.value(x)

        return out


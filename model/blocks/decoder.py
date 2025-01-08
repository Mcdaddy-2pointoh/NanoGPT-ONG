import torch
import torch.nn as nn
from torch.nn import functional as F
from model.layers.MultiheadMaskedAttention import MultiHeadAttention
from model.layers.FeedForwardNetwork import FeedForward
import warnings

class Block(nn.Module):
    """
    Class: One single transformer block that can be replicated
    """

    def __init__(self, 
                 num_heads: int, 
                 block_size: int, 
                 n_embedd: int = 32, 
                 device: str = None, 
                 attention_size: int = 16, 
                 dropout: float = 0.2,
                 positional_encoder_type: str = None,
                 model_precision = torch.float32
                 ):
        """
        Function: Instances an object of class `MultiHeadAttention`
        Args:
            num_heads (int): Number of parallel heads to implement
            block_size (int): Block size is the maximum context window of the model
            n_embedd (int): Linear dimension in which the token in projected into
            device (str): The device on which the operation must be carried out `cuda` or `cpu`
            attention_size (int): The projection dimension of all attention heads combined
            dropout (float): The value of dropout to be added at end of each layer
            model_precision : Define the model float precision
            positional_encoder_type (Enum(str)): Either has conventional linear postional encoding, RoPE or sinusoidal positional encoding
        """
        super().__init__()

        
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

        # Set model precision to default with a warning
        if model_precision not in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
            warnings.warn("Defaulting to torch.float32, {model_precision} is not a valid dtype")
            self.model_precision = torch.float32

        else:
            self.model_precision = model_precision

        model_precision = self.model_precision

        # Attention params and methods
        individual_head_size = attention_size//num_heads
        self.individual_head_size = individual_head_size
        self.attention_head = MultiHeadAttention(num_heads = num_heads, n_embedd=n_embedd, block_size=block_size, device=device, attention_head_size=individual_head_size, dropout = dropout, positional_encoder_type=positional_encoder_type, model_precision=model_precision)
        
        # Feed forward params and methods
        self.ffn = FeedForward(attention_size=attention_size, dropout = dropout, device=device, model_precision=model_precision)

        # Layer normalisation
        self.ln1 = nn.LayerNorm(attention_size).to(device=device, dtype=model_precision)
        self.ln2 = nn.LayerNorm(attention_size).to(device=device, dtype=model_precision)

    def forward(self, x):
        """
        Function: Feed forward function of the model 
        Args:
            x (torch.Tensor): Input params of the model 
        """
        x = x.to(dtype=self.model_precision)
        x = x + self.attention_head(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x
        
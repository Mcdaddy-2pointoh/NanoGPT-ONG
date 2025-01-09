import torch 
from torch import nn
from torch.nn import functional as F
import warnings

class FeedForward(nn.Module):
    """
    A single MLP followed by non-linearity
    """

    def __init__(self, attention_size: int = 16, dropout: float = 0.2, device: str=None, model_precision = torch.float32):
        """
        Function: initialises a simple feed forward nn
        Args:
            attention_size (int): The projection dimension of all attention heads combined
            dropout (float): The value of dropout to be added at end of each layer
            device (str): The device on which the operation must be carried out `cuda` or `cpu`
            model_precision : Define the model float precision
        """
        
        super().__init__()

        # Validate attention head size 
        try:
            attention_size = int(attention_size)
        except Exception as e:
            raise TypeError("The argument `attention_size` must be of type int.") from e

        # Determine the device type
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device in ["cuda", "cpu"]:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        device = self.device

        # Set model precision to default with a warning
        if model_precision not in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
            warnings.warn("Defaulting to torch.float32, {model_precision} is not a valid dtype")
            self.model_precision = torch.float32

        else:
            self.model_precision = model_precision

        model_precision = self.model_precision

        self.net = nn.Sequential(
            nn.Linear(attention_size, attention_size * 4),
            nn.GELU(),
            nn.Linear(attention_size * 4, attention_size),
            nn.Dropout(dropout)
        ).to(device=device, dtype=model_precision)

    def forward(self, x):
        return self.net(x)
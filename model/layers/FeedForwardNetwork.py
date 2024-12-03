import torch 
from torch import nn
from torch.nn import functional as F

class FeedForward(nn.Module):
    """
    A single MLP followed by non-linearity
    """

    def __init__(self, attention_size: int = 16, dropout: float = 0.2, device: str=None):
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

        self.net = nn.Sequential(
            nn.Linear(attention_size, attention_size * 4),
            nn.GELU(),
            nn.Linear(attention_size * 4, attention_size),
            nn.Dropout(dropout)
        ).to(device=device)

    def forward(self, x):
        return self.net(x)
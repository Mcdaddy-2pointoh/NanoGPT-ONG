import torch 
from torch import nn
from torch.nn import functional as F

class FeedForward(nn.Module):
    """
    A single MLP followed by non-linearity
    """

    def __init__(self, attention_head_size: int = 16, device: str=None):
        super().__init__()

        # Validate attention head size 
        try:
            attention_head_size = int(attention_head_size)
        except Exception as e:
            raise TypeError("The argument `attention_head_size` must be of type int.") from e
        

        # Determine the device type
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device in ["cuda", "cpu"]:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        device = self.device

        self.net = nn.Sequential(
            nn.Linear(attention_head_size, attention_head_size),
            nn.ReLU()
        ).to(device=device)

    def forward(self, x):
        return self.net(x)
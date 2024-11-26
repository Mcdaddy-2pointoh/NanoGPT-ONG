import torch
from torch import nn

def SinusoidalPositionalEncoding(T: int, n_embedd: int, n: int = 10_000, device: str = "cpu"):
    """
    Function: Retrieves Sinusoidal Positional encoding for various dimensions
    Args:
        x (torch.Tensor): The vector to be positionally encoded
        n_embedd (int): Linear dimension in which the token in projected into
        device (str): The device on which the operation must be carried out `cuda` or `cpu`
    """
    
    # If the n_embedd is odd raise error
    if n_embedd % 2 != 0:
        raise ValueError(f"Argument `n_embedd` must be an even number for Sinusoidal Positional Encoding")
    
    # Initialise embedding map 
    positions = torch.arange(0, T).unsqueeze_(1).to(device=device)
    embeddings = torch.zeros(T, n_embedd).to(device=device)

    # Calculate denominator map
    denom = torch.pow(n, 2*torch.arange(0, n_embedd//2)/n_embedd).to(device=device)

    # Update embedding
    embeddings[:, 0::2] = torch.sin(positions/denom).to(device=device)
    embeddings[:, 1::2] = torch.cos(positions/denom).to(device=device)

    # Get embedding map
    return torch.tensor(embeddings).to(device=device)

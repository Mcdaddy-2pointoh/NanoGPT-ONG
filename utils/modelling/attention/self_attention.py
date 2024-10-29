import torch
import torch.nn.functional as F

# Version 1: Averaging
def self_attention_v1(x):
    """
    Function: Add some form of communication between previous tokens and the current nth token by creating a rollup on the previous tokens and all channels
    Args:
        x (tensor): A tensor with 3 dimesions i.e. batch, timestep, channels
    """

    # Determine the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert x
    if type(x) != torch.Tensor:
        x = torch.tensor(x).to(device=device)
    
        if not len(x.shape) == 3:
            raise ValueError("Expected tensor of 3 dimensions")
        
    else:
        x = x.to(device=device)

    # Initialise an empty tensor
    x_bow = torch.zeros(size=x.shape).to(device=device)

    # Average the previous inputs over time
    for batch_idx, batch in enumerate(x):
        for time_index, time in enumerate(batch):
            x_previous = x[batch_idx, :time_index+1]
            x_bow[batch_idx, time_index] = torch.mean(x_previous, 0)
    return x_bow

# Version 2: MatMul Averaging
def self_attention_v2(x):
    """
    Function: Use matrix multiplication for the self_attention
    Args:
        x (tensor): A tensor with 3 dimesions i.e. batch, timestep, channels
    """   

    # Determine the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert x
    if type(x) != torch.Tensor:
        x = torch.tensor(x).to(device=device)
    
        if not len(x.shape) == 3:
            raise ValueError("Expected tensor of 3 dimensions")
        
    else:
        x = x.to(device=device)

    # Get shape of x
    B, T, C = x.shape

    # Generate a weight matrix for multiplication
    wei = torch.tril(torch.ones(T, T)).to(device=device)
    wei = wei / wei.sum(1, keepdim=True)

    # Stacking the weighted matrix B times to replicate operation for all matrix batches
    wei = torch.stack([wei]*B).to(device=device)

    # Multiplying the weight through all the batches of x
    x_bow_2 = wei @ x

    return x_bow_2


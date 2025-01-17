# Imports
import torch
from data_processing.augmenters import batch_generator
from tqdm import tqdm


def naive_trainer(data: torch.Tensor, model: torch.nn.Module, optimizer, batch_size: int, block_size: int, steps: int = 100):
    """
    Function: Basic training loop that trains a pytorch module using an optimizer and batch size for n-steps
    Args:
        data (torch.Tensor): Tokenized data array
        model (torch.nn.Module): Decoder only transformer model to train
        optimizer : Torch optimizer object used for training the model
        block_size (int): Block size is the maximum context window of the model
        batch_size (int): Batch size is the number of concurrent samples we can load for GPU saturation
        steps (int) : The number of steps to train the model for
    """
    # Check device 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Validate dtypes
    if type(data) != torch.Tensor:
        try:
            data = torch.tensor(data=data)
        except Exception as e:
            raise ValueError(f"Could not convert data of type {type(data)} to torch.Tensor")
    
    elif not isinstance(model, torch.nn.Module):
        raise TypeError("Model must be an instance of torch.nn.Module")
    
    elif not isinstance(steps, int):
        try:
            steps = int(steps)
        except Exception as e:
            raise ValueError("Steps must by of type `int`, could not covenrt steps to type int") from e
    
    losses_at_step = []

    for step in tqdm(range(steps)):

        # Sample a batch of data
        try:
            xb, yb = batch_generator(data=data, block_size=block_size, batch_size=batch_size, as_torch_tensors=True, device=device)
        except Exception as e:
            raise RuntimeError("Could not generate a batch from data.") from e
        
        # Evaluate the loss 
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses_at_step.append(loss.item())

    return model, losses_at_step   
        
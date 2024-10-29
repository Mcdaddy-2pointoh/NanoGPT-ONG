# Imports
import torch
from utils.data.augmenters import batch_generator
from tqdm import tqdm
def naive_trainer(data: torch.Tensor, model: torch.nn.Module, optimizer, batch_size: int, block_size: int, steps: int = 100):
    
    # Check device 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

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
        
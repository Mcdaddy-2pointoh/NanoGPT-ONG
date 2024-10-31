from matplotlib import pyplot as plt
import os
import numpy as np

def plot_loss(losses, dir_path, smoothen=False):
    """
    Function: Plots and saves the loss as an image per run
    Args:
        losses(list | torch.Tensor) : list of loss per step
        dir_path(str) : to save the image 
    """
    # Check if directory exists at path 
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"No directory found at {dir_path}")
    else:
        run_number = str(len(os.listdir(path=dir_path)) + 1).zfill(4)

    # If smoothen
    if smoothen:
        losses = estimate_loss(losses=losses)

    try: 
        plt.plot(list(range(1,len(losses)+1)), losses)
        plt.savefig(os.path.join(dir_path, f"loss-run-{run_number}.png"))

    except Exception as e:
        raise RuntimeError("Could not save loss curve") from e
    
def estimate_loss(losses, skip: int = None):
    """
    Function: Smoothens the losses over `skip` steps
    Args:
        losses(list | torch.Tensor) : list of loss per step
        skip (None | int): Smoothening window
    """
    if skip is None:
        skip = int(len(losses)/100)

    ret = np.cumsum(losses, dtype=float)
    ret[skip:] = ret[skip:] - ret[:-skip]
    return ret[skip - 1:] / skip
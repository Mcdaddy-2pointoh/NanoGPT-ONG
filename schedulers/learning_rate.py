from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from schedulers.validate import validate_cosine_annealing, validate_cosine_annealing_warm_restarts

# Function to initialise a scheduler object
def set_lr_scheduler(optimizer, scheduler_type: str, scheduler_params: dict):
    """
    Function: To initialise a LR scheduler object for training
    Args:
        optimizer (torch.optim): Optimizer object to initialise the scheduler for
        scheduler_type (str): The name of the schduler to be initialised ["cosine", "CosineAnnealingWarmRestarts"]
        scheduler_params (dict): The arguments for scheduler
    """

    if scheduler_type == "CosineAnnealingLR":

        # Validate and initialise a lr_scheduler object
        try:
            # Get scheduler params 
            # T_max (int) – Maximum number of iterations
            # eta_min (float) – Minimum learning rate. Default: 0
            # last_epoch (int) – The index of last epoch. Default: -1

            T_max, eta_min, last_epoch = validate_cosine_annealing(scheduler_params=scheduler_params)

            # Return a scheduler object
            return CosineAnnealingLR(optimizer=optimizer, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch)

        except Exception as e:
            raise RuntimeError('Failed to initialise a CosineAnnealingLR Scheduler object.')
            

    elif scheduler_type == "CosineAnnealingWarmRestarts":

        # Validate and initialise a lr_scheduler object for CosineAnnealingWarmRestarts
        try:
            # Get scheduler params
            # T_0 (int) – Number of iterations until the first restart
            # T_mult (int, optional) – A factor by which Ti increases after restart. Default 1
            # eta_min (float, optional) – Minimum learning rate. Default: 0
            # last_epoch (int, optional) – The index of the last epoch. Default: -1
            T_0, T_mult, eta_min, last_epoch = validate_cosine_annealing_warm_restarts(scheduler_params=scheduler_params)

            # Return a scheduler object
            return CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min, last_epoch=last_epoch)
        
        except Exception as e:
            raise RuntimeError('Failed to initialise a CosineAnnealingLR Scheduler object.')

    else:
        raise ValueError(f"No LR scheduler named {scheduler_type} found")
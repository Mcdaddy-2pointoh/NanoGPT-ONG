
def validate_cosine_annealing(scheduler_params: dict):
    """
    Function: That validates the scheduler params for cosine_annealing scheduler
    Args:
        scheduler_params (dict): The dictionary containing all params
    """

    # PARAM VALIDATION
    # Validate the params dictionary and return the param args
    # Use default args for optional params
    try:

        # Set the `T_max` params
        if 'T_max' not in scheduler_params.keys():
            raise KeyError("Could not initialise `CosineAnnealingLR` object. Missing key `T_max` in `scheduler_params`")

        # Set the eta_min param
        if 'eta_min' in scheduler_params.keys():
            eta_min = scheduler_params['eta_min']

        else:
            eta_min = 0

        # Set the last_epoch param
        if 'last_epoch' in scheduler_params.keys():
            last_epoch = scheduler_params['last_epoch']

        else:
            last_epoch = -1

        # Validate the scheduler params
        return scheduler_params['T_max'], eta_min, last_epoch

    # Raise exception
    except Exception as e:
        raise RuntimeError("Could not initialise CosineAnnealing LR.") from e
    
def validate_cosine_annealing_warm_restarts(scheduler_params: dict):
    """
    Function: That validates the scheduler params for cosine_annealing scheduler
    Args:
        scheduler_params (dict): The dictionary containing all params
    """

    # PARAM VALIDATION
    # Validate the params dictionary and return the param args
    # Use default args for optional params if not provided
    try:
        
        # Validate the T_0 param
        if 'T_0' not in scheduler_params.keys():
            raise KeyError("Could not initialise `CosineAnnealingWarmRestarts` object. Missing key `T_0` in `scheduler_params`")
        
        # Validate the T_mult param
        if 'T_mult' not in scheduler_params.keys():
            T_mult = 1

        else: 
            T_mult = scheduler_params['T_mult']

        # Set the eta_min param
        if 'eta_min' in scheduler_params.keys():
            eta_min = scheduler_params['eta_min']

        else:
            eta_min = 0

        # Set the last_epoch param
        if 'last_epoch' in scheduler_params.keys():
            last_epoch = scheduler_params['last_epoch']

        else:
            last_epoch = -1

        # Validate the scheduler params
        return scheduler_params['T_0'], T_mult, eta_min, last_epoch 

    except Exception as e:
        raise RuntimeError("Could not initialise `CosineAnnealingWarmRestarts` LR.") from e
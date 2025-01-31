from data_processing.augmenters import file_splitter, segmented_tokenization
import os
import torch
from tqdm import tqdm
from tokenizers.tiktokenizer import tiktokenizer
from trainers.lazy_batch_trainer.trainer import lazy_batch_trainer
from model.models import LanguageModel
import numpy as np
import json
from schedulers.learning_rate import set_lr_scheduler
from telemetry.visualisers import plot_loss, plot_lr

def continued_pretrainer(
    model_checkpoint_path: str,
    optimizer_checkpoint_path: str,
    params_path: str,
    training_params: dict,
    array_directory: str,
    check_point_params: dict,
    runs_dir: str = "./runs",
    device: str = None,
    smoothen_loss_plots: bool = True,
    save_loss_curves: bool = True
):
    
    """
    Fucntion: Can load and train a model from a checkpoint specified, needs pretokenized data
    Args: 
        model_checkpoint_path (str): The path to the model checkpoint 
        optimizer_checkpoint_path (str): The path to the optimizer checkpoint
        params_path (str): The path to the model params & metadata file
        device (str): The compuute unit to use
    """


    # CREATE A RUN DIRECTORY
    # Logs the loss curve
    # Logs the model learning rate
    # Logs the model state dictionary 
    # Logs the optimizer state dictionary
    if not os.path.exists(runs_dir):
        os.mkdir(runs_dir)
        print(f"Making a directory `runs` in {os.getcwd()}")
        run_number = str(1).zfill(4)
    
    # Create run number
    else:
        run_number = str(len(os.listdir(runs_dir)) + 1).zfill(4)

    # Create a run directory
    os.mkdir(f"{runs_dir}/run-{run_number}")
    os.mkdir(f"{runs_dir}/run-{run_number}/metric-logs")
    os.mkdir(f"{runs_dir}/run-{run_number}/results")
    os.mkdir(f"{runs_dir}/run-{run_number}/model")
    os.mkdir(f"{runs_dir}/run-{run_number}/tokenizer")
    os.mkdir(f"{runs_dir}/run-{run_number}/checkpoints")

    # Validate the array_directory 
    if not os.path.exists(array_directory):
        raise FileNotFoundError("Could not locate the array directory containing pretokenized data arrays with `.npy` format")
    
    elif len([i for i in os.listdir(array_directory) if i.endswith(".npy")]) == 0:
        raise FileNotFoundError("Found empty directory with no `.npy files`")
    
    
    # LOAD THE PARAMS file
    # Use the model configs to init a model
    with open(params_path) as f:
        model_params = json.load(f)
    
    # SET THE DEVICE
    # Get a list of all possible devices
    devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())] + ['cpu']

    # If the chosen device is not in the torch available device just print a warning and default to the first available device
    if device not in devices:
        device = devices[0]
    

    # LOAD THE TOKENIZERS
    # Specify the tokenizers type 
    # Load the tokenizer from tiktokens
    print("LOADING TOKENIZER")
    tokenizer = tiktokenizer(encoding=model_params['tokenizer_encoding'])


    # LOAD THE MODEL
    # Specify the model path
    # Sepcify the model param file 
    # Load the model in train mode

    # Initialise the model structure using the params file 
    try:
        print("LOADING MODEL")
        model = LanguageModel(vocab_size=model_params['vocab_size'], 
                                        block_size=model_params['block_size'], 
                                        n_embedd=model_params['n_embedd'],
                                        device=device,
                                        attention_size=model_params['attention_size'],
                                        num_heads= model_params['num_heads'],
                                        num_layers=model_params['num_layers'],
                                        model_precision=model_params['model_precision'],
                                        dropout = 0,
                                        positional_encoder_type=model_params["positional_encoder_type"]
                                        )
        
        # Load the models weights into the abovde defined architecture
        model.load_state_dict(torch.load(model_checkpoint_path, weights_only=True, map_location=device))
        model = model.to(device=device)
        model.train()

    except Exception as e:
        raise RuntimeError("Failed to initialise a `Optimizer` object from the state_dict") from e

    # VALIDATE TRAINING PARAMS
    # Valdiate dtype learning_rate
    # Valdiate dtype batch_size
    # Valdiate dtype steps
    print("VALIDATE TRAINING PARAMS")
    if not isinstance(training_params, dict):
        raise ValueError("Argument `training_params` must be of type dict.")

    elif not set(list(training_params.keys())).issuperset(set(['learning_rate', 'batch_size', 'steps'])):
        raise ValueError("Argument `training_params` must be of type dict and must have keys ['learning_rate', 'batch_size', 'steps']")
    
    # Validate the learning_rate
    elif not isinstance(training_params['learning_rate'], float):
        try:
            training_params['learning_rate'] = float(training_params['learning_rate'])

        except Exception as e:
            raise TypeError("Argument `training_params['learning_rate']` must be of type `int`") from e
        
    # Validate the batch_size
    elif not isinstance(training_params['batch_size'], int):
        try:
            training_params['batch_size'] = int(training_params['batch_size'])

        except Exception as e:
            raise TypeError("Argument `training_params['batch_size']` must be of type `int`") from e
        
    # Validate the steps
    elif not isinstance(training_params['steps'], int):
        try:
            training_params['steps'] = int(training_params['steps'])

        except Exception as e:
            raise TypeError("Argument `training_params['steps']` must be of type `int`") from e  
        
    else:

        # LOAD THE OPTIMIZER
        # Create a base instance of the optimizer
        # Restore optimizer state using load_state_dict 
        # Add a scheduler if there is a LR scheduling instance 
        try: 
            print("LOADING OPTIMIZER")
            optimizer = torch.optim.AdamW(model.parameters(), lr=training_params['learning_rate'])

            # Load the checkpoint params
            optimizer.load_state_dict(torch.load(optimizer_checkpoint_path, weights_only=True,map_location=device))

            if 'lr_scheduler_type' in training_params.keys() and training_params['lr_scheduler_type'] is not None:
                lr_scheduler = set_lr_scheduler(optimizer, training_params['lr_scheduler_type'], training_params['lr_scheduler_params'])

            else:
                lr_scheduler = None

        except Exception as e:
            raise RuntimeError("Failed to initialise a `Optimizer` object from the state_dict") from e
        

        # MODEL TRAINING
        # Specify the checkpoiniting dir
        # Call the lazy batch trainer function
        # Save the model params
        check_point_params['save_dir'] = f"{runs_dir}/run-{run_number}/checkpoints"

        # Parsing the model and optimizer to the training theory
        model, losses, lr = lazy_batch_trainer(dir_path=array_directory,
                                        model=model,
                                        optimizer=optimizer,
                                        batch_size=training_params['batch_size'],
                                        block_size=model_params['block_size'],
                                        steps=training_params['steps'],
                                        device=device,
                                        check_point_params=check_point_params,
                                        train_ratio=0.95,
                                        model_params=model_params,
                                        lr_scheduler=lr_scheduler
                                        )
        
        # Save loss to a directory
        if save_loss_curves:
            plot_loss(losses, f"{runs_dir}/run-{run_number}/metric-logs", smoothen=smoothen_loss_plots)
            plot_lr(lr, f"{runs_dir}/run-{run_number}/metric-logs", smoothen=smoothen_loss_plots)

        torch.save(model.state_dict(), f"{runs_dir}/run-{run_number}/model/LanguageModel.pt")
        torch.save(optimizer.state_dict(), f"{runs_dir}/run-{run_number}/model/Optimizer.pt")
        with open (f"{runs_dir}/run-{run_number}/model/params.json", "w") as f:
            json.dump(model_params, f)

        # Save the losses to the npy file
        losses = np.array(losses)
        lr = np.array(lr)
        np.save(f"{runs_dir}/run-{run_number}/metric-logs/losses.npy", losses)
        np.save(f"{runs_dir}/run-{run_number}/metric-logs/lr.npy", lr)

        return model, losses, lr

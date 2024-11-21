import torch
import os
from tqdm import tqdm
from utils.modelling.trainers.lazy_batch_trainer.switch import switch
import numpy as np
from utils.data.augmenters import train_test_splitter, batch_generator

def lazy_batch_trainer(dir_path: str, model: torch.nn.Module, optimizer, batch_size: int, block_size: int, steps: int, check_point_params: dict = None, device: str = 'cpu', train_ratio: float = 0.90):
    """
    Function: Trains the model in parts upon each of the file in dir_path
    Args:
        dir_path (str): The directory containing all segmented text files
        model (torch.nn.Module): Decoder only transformer model to train
        optimizer : Torch optimizer object used for training the model
        block_size (int): Block size is the maximum context window of the model
        batch_size (int): Batch size is the number of concurrent samples we can load for GPU saturation
        steps (int) : The number of steps to train the model for 
        check_pointing (dict): Dictionary contains information about the checkpointing the model
        device (str) : Set the computational device  
        train_ratio (0 < float < 1): The ratio of data to be considered as training data
    """

    # Create a dictionary to validate the coverage of data used
    if not os.path.exists(dir_path):
        raise ValueError("Argument `dir_path` must be a path to a directory containing .npy files")
    
    else:

        # validate files in the directory
        array_files = os.listdir(dir_path)
        array_files = [array_file for array_file in os.listdir(dir_path) if array_file.endswith(".npy")]

        # Raise error if no .npy file is present in the directory
        if len(array_files) == 0:
            raise ValueError("No .npy file found inf the `dir_path` provided")
        
        # Else create a status dictionary
        # Monitors which files have been trained upon 
        # What files are pending to be trained upon
        # losses in each file
        else:

            # Keeps track of training telemetry
            status_dictionary = {
                "pending" : array_files,
                "trained" : [],
                "losses" : {},
                # "segment_utilization" : To be coded
                }
            
            # Total segment files
            total_segments = len(array_files)

            # Change frequency 
            change_frequency =  steps // total_segments


            # Inital params 
            loss = None
            cummulated_loss = []
            training_segment = None
            checkpoint_idx = str(1).zfill(5)

            # Checkpointing validation
            if check_point_params is not None:                

                # Validate the path and the step frequency
                if not isinstance(check_point_params, dict):
                    raise TypeError("Argument `check_point_params` must be of type dict")
                
                # Raise error if key 'save_steps', 'save_dir' aren't in the params
                elif not set(list(check_point_params.keys())).issuperset(set(['save_steps', 'save_dir'])):
                    raise ValueError("Argument `check_point_params` must have keys ['save_steps', 'save_dir']")
                
                # Raise error if save steps is not a int or is greater than the steps
                elif not isinstance(check_point_params['save_steps'], int) or check_point_params['save_steps'] > steps:
                    raise TypeError("Argument `check_point_params['save_steps']` must be of type int less than the training steps") 
                
                # Raise error if save steps is not a int or is greater than the steps
                elif not isinstance(check_point_params['save_dir'], str):
                    raise TypeError("Argument `check_point_params['save_dir']` must be of type str and a path to save the model checkpoints") 
                
                # Validate train ratio
                elif not isinstance(train_ratio, float) or train_ratio > 1 or train_ratio < 0:
                    raise ValueError("Argument `train_ratio` must be a float between 0 and 1.")
                
                # Make a directory if it does not exist
                elif not os.path.isdir(check_point_params['save_dir']):
                    try: 
                        os.makedirs(check_point_params['save_dir'])

                    except Exception as e:
                        raise RuntimeError("Could not create checkpoint directories") from e

            for step in tqdm(range(steps)):
                
                # Select a new segment to train upon
                if step % change_frequency == 0:

                    # Updated status dictionary
                    status_dictionary, training_segment = switch(status_dictionary=status_dictionary,
                                              loss=loss,
                                              previous_segment=training_segment)
                    
                    print(training_segment)

                    # Training Segment is None implying no more data left
                    if training_segment is None:
                        break
                    
                    # Set new training loss to data
                    loss = []

                    # Load the numpy array
                    data_path = os.path.join(dir_path, training_segment)
                    data_arr = list(np.load(data_path))

                    # Create train test split
                    train_data, test_data = train_test_splitter(data=data_arr, split_ratio=train_ratio)

                # Checkpoint the model
                if check_point_params is not None and (step % check_point_params['save_steps'] == 0 and not step == 0):

                    # Validation loss 
                    model.eval()

                    # Save model and optimizers
                    torch.save(model.state_dict(), f"{check_point_params['save_dir']}/LanguageModel-checkpoint-{checkpoint_idx}.pt")
                    torch.save(optimizer.state_dict(), f"{check_point_params['save_dir']}/Optimizer-checkpoint-{checkpoint_idx}.pt")

                    model.train()

                    # Set checkpoint to next idx
                    checkpoint_idx = str(int(checkpoint_idx) + 1).zfill(5)

                # Get batch from batch loader
                try:
                    xb, yb = batch_generator(data=train_data, block_size=block_size, batch_size=batch_size, as_torch_tensors=True, device=device)
                    xb, yb  = xb.to(device=device), yb.to(device=device)

                except Exception as e:
                    raise RuntimeError("Could not generate a batch from data.") from e
                
                # Train the model
                logits, batch_loss = model(xb, yb)
                optimizer.zero_grad(set_to_none=True)
                batch_loss.backward()
                optimizer.step()
                loss.append(batch_loss.item())
                cummulated_loss.append(batch_loss.item())

            return model, cummulated_loss

                



                    

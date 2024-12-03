from data_processing.augmenters import file_splitter, segmented_tokenization
import os
import torch
from tqdm import tqdm
from tokenizers.tiktokenizer import tiktokenizer
from trainers.lazy_batch_trainer.trainer import lazy_batch_trainer
from model.models import LanguageModel
import numpy as np
import json
from telemetry.visualisers import plot_loss


def lazy_batch_training(
    data: str,
    file_splitter_params: dict,
    tokenizer_encoding: str,
    tokenizer_vocab_size: int,
    model_params: dict,
    training_params: dict,
    check_point_params: dict,
    segment_data: bool = True,
    runs_dir: str = "./runs",
    device: str = "cpu",
    smoothen_loss_plots: bool = True,
    save_loss_curves: bool = True
):
    """
    Function: A Pipeline that executes a trainer that can segment a large training text file into multiple smaller text files and saves it. 
            The pipeline then pre-tokenzes the data and stores it into the secondary memory. The training loop sequentially picks up each tokenized segment and train the model.
            The pipeline also has checkpointing feature and loss eval too.

    Args:
        data (str): Path to the single text file
        file_splittter_params (dict): Dictionary defining the parameters to spilt the files into smaller segments. Contains keys ['segment_target_dir', 'array_target_dir',  'split_threshold', 'verbose', 'file_encoding', 'write_frequency']
        tokenizer_encoding (str): The class of BPE tokenizer to use from tiktoken
        tokenizer_vocab_size (int): Number of unique tokens in the `tokenizer_encoding`
        model_params (dict): Dictionary defining the model architecture and layer sizes, for the decoder only language model. Contains keys ['block_size', 'n_embedd', 'attention_size', 'num_heads', 'num_layers', 'dropout', 'positional_encoder_type']
        training_param (dict): Dictionary defining the training setup of the model. Contains keys ['learning_rate', 'batch_size', 'steps']
        runs_dir (str): Path to the checkpointing directory
        device (str): The device to perfrom compute on 
        smoothen_loss_plots (bool): Smoothen the loss curve while plotting the train loss
        save_loss_curves(bool): Saves the loss curve as a png in the checkpointing directory
    """
    
    # SETTING UP RUNS
    # Create a directory to store all runs metadata
    # Validate compute devices available on the PC
    # validate if the run directory exists else make one & create a run number 
    # Define the compute unit to use

    # Check device 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(runs_dir):
        os.mkdir(runs_dir)
        print(f"Making a directory `runs` in {os.getcwd()}")
        run_number = str(1).zfill(4)
    
    # Create run number
    else:
        run_number = str(len(os.listdir(runs_dir)) + 1).zfill(4)

    # Create a run directory
    os.mkdir(f"{runs_dir}/run-{run_number}")
    os.mkdir(f"{runs_dir}/run-{run_number}/loss-logs")
    os.mkdir(f"{runs_dir}/run-{run_number}/results")
    os.mkdir(f"{runs_dir}/run-{run_number}/model")
    os.mkdir(f"{runs_dir}/run-{run_number}/tokenizer")
    os.mkdir(f"{runs_dir}/run-{run_number}/checkpoints")

    # Check device 
    device = "cuda" if torch.cuda.is_available() else "cpu"


    # DATA INGESTION 
    # Validate the data input
    # Segment the files if specified
    # Save the data into smaller files
    # Validate the input

    if not isinstance(data, str):
        raise ValueError("Argument data must be the path to the .txt file containing the data, Invalid data type found")
    
    elif not os.path.isfile(data):
        raise ValueError("Argument data must be the path to the .txt file containing the data, Invalid file path")
    
    elif data.split(".")[-1] != "txt":
        raise TypeError(f"Path `{data}` must be of type .txt")
    
    # Segmenting files if all input is correct
    else:
        # If segment_data is true split text into smaller files
        if segment_data:

            # Validate file_splitter_params
            if not set(list(file_splitter_params.keys())).issuperset(set(['segment_target_dir', 'array_target_dir', 'split_threshold', 'verbose', 'file_encoding', 'write_frequency'])):
                raise ValueError("Argument `file_splitter_params` must be a dictionary with the following keys ['segment_target_dir', 'array_target_dir',  'split_threshold', 'verbose', 'file_encoding', 'write_frequency']")
 
            # Invalid file encoding format
            elif not isinstance(file_splitter_params['file_encoding'], str):
                raise TypeError(f"Argument `file_encoding` must be of type string and a valid file encoding format")
            
            # Validate target_dir
            elif not os.path.isdir(file_splitter_params['segment_target_dir']):
                raise FileNotFoundError(f"Argument `target_dir` must be the path to an existing file.")
            
            # Split threshold is not an int
            elif not isinstance(file_splitter_params['split_threshold'], int):
                raise TypeError(f"Argument `split_threshold` must be of type int.")

            # Target directory is not empty
            elif len(file_splitter_params['segment_target_dir']) != 0:
                value = str(input("File dir is not empty do you want to overwrite existing file `Y / N`?")).upper()

                if value not in ['Y', 'N']:
                    raise ValueError("Invalid value typed in expected str 'Y' or 'N'")
                
                elif value == "Y":
                    try:
                        # Clean the directory 
                        files = os.listdir(file_splitter_params['segment_target_dir'])
                        for file in files:
                            file_path = os.path.join(file_splitter_params['segment_target_dir'], file)
                            if os.path.isfile(file_path):
                                os.remove(file_path)

                        # Run partitioning
                        print("Please monitor CPU stats while splitting files")
                        file_splitter(
                            data=data,
                            target_dir=file_splitter_params['segment_target_dir'],
                            split_threshold=file_splitter_params['split_threshold'], # 200k lines per file segment
                            write_frequency=file_splitter_params['write_frequency'],
                            file_encoding=file_splitter_params['file_encoding'],
                            verbose=file_splitter_params['verbose'])  
                    
                    except Exception as e:
                        raise RuntimeError("Could not split file into segments") from e
                
                else:
                    value = str(input("Do you want to segment the new data `Y / N`?")).upper()

                    # Partition data without overwriting
                    if value == "Y":
                        try:
                            print("Please monitor CPU stats while splitting files")
                            file_splitter(
                                    data=data,
                                    target_dir=file_splitter_params['segment_target_dir'],
                                    split_threshold=file_splitter_params['split_threshold'], # 200k lines per file segment
                                    write_frequency=file_splitter_params['write_frequency'],
                                    file_encoding=file_splitter_params['file_encoding'],
                                    verbose=file_splitter_params['verbose'])  
                            
                        except Exception as e:
                            raise RuntimeError("Could not split file into segments") from e
                        
                    # Skipping partioning
                    elif value == "N":
                        data_dir = file_splitter_params['segment_target_dir']
                        print("Skipped Partitioning")

                    # Raise error for incorrect input
                    else:
                        raise ValueError("Invalid value typed in expected str 'Y' or 'N'")

            # Target directory is empty just begin partitioning
            else:
                try:
                    print("Please monitor CPU stats while splitting files")
                    data_dir = file_splitter(
                            data=data,
                            target_dir=file_splitter_params['segment_target_dir'],
                            split_threshold=file_splitter_params['split_threshold'], # 200k lines per file segment
                            write_frequency=file_splitter_params['write_frequency'],
                            file_encoding=file_splitter_params['file_encoding'],
                            verbose=file_splitter_params['verbose'])      

                except Exception as e:
                    raise RuntimeError("Could not split file into segments") from e            
        
        # Just ignore processing
        else:
            data_dir = os.path.dirname(data)
    
    # DATA TOKENIZATION
    # Load the data from each segment file
    # tokenization can only be done via tik-tokens no naive token
    
    tokenizer = tiktokenizer(encoding=tokenizer_encoding)

    # If target directory does not exist make one
    if not os.path.isdir(file_splitter_params['array_target_dir']):
        try:

            # Make directory
            os.makedirs(file_splitter_params['array_target_dir'])
           
            # Run batched tokenization 
            print("Please monitor CPU stats while tokenizing files")
            segmented_tokenization(data_dir=data_dir, tokenizer=tokenizer, file_encoding=file_splitter_params['file_encoding'], target_dir=file_splitter_params['array_target_dir'])

        except Exception as e:
            raise RuntimeError("Could not create directories") from e
        

    # Else if the directory has some arrays already present 
    elif os.path.isdir(file_splitter_params['array_target_dir']) and len(os.listdir(file_splitter_params['array_target_dir'])) != 0:

        # Ask if we want to clean the directory 
        value = str(input("The directory seems to have some array files in them, do you want to overwrite on them `Y / N`.")).upper()

        # If overwrite is permitted '
        # Wipe directory and store the tokenized array as npy file
        if value == "Y":

            try:
                # Clean the directory 
                files = os.listdir(file_splitter_params['array_target_dir'])
                for file in files:
                    file_path = os.path.join(file_splitter_params['array_target_dir'], file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

                # Perform segment tokenization
                print("Please monitor CPU stats while tokenizing files")
                array_directory = segmented_tokenization(
                    data_dir=data_dir,
                    tokenizer=tokenizer,
                    file_encoding=file_splitter_params['file_encoding'],
                    target_dir=file_splitter_params['array_target_dir']
                )
            
            except Exception as e:
                raise RuntimeError("Could not create array files") from e

        elif value == "N":

            # Do not clean the directory but ask if 
            # we need to perform segmentation again

            value = str(input("Do we need to re-tokenized the segmented files `Y / N`.")).upper()

            # Re-Tokenize the data into the array files
            if value == "Y":

                try:
                    print("Please monitor CPU stats while tokenizing files")
                    array_directory = segmented_tokenization(
                        data_dir=data_dir,
                        tokenizer=tokenizer,
                        file_encoding=file_splitter_params['file_encoding'],
                        target_dir=file_splitter_params['array_target_dir']
                    )

                except Exception as e:
                    raise RuntimeError("Could not create array files") from e


            # Just return the directory name
            elif value == "N":
                array_directory = file_splitter_params['array_target_dir']

    # Else if there is a directory just perform tokenization
    else:
        try:
            print("Please monitor CPU stats while tokenizing files")
            array_directory = segmented_tokenization(
                data_dir=data_dir,
                tokenizer=tokenizer,
                file_encoding=file_splitter_params['file_encoding'],
                target_dir=file_splitter_params['array_target_dir']
            )

        except Exception as e:
            raise RuntimeError("Could not create array files") from e

    # MODEL TRAINING
    # Given N files and M training steps 
    # Train the model for M//N steps on a single array file

    # Validate input params
    if not isinstance(tokenizer_vocab_size, int):
        raise TypeError("Argument `tokenizer_vocab_size` must be of type int")
    
    # Validate model_params
    elif not isinstance(model_params, dict):
        raise TypeError("Argument `model_params` must be of type dict")

    elif not set(list(model_params.keys())).issuperset(set(['block_size', 'n_embedd', 'attention_size', 'num_heads', 'num_layers', 'dropout', 'positional_encoder_type'])):
        raise ValueError("Argument `model_params` must be a dictionary with the following keys 'block_size', 'n_embedd', 'attention_size', 'num_heads', 'num_layers', 'dropout', 'positional_encoder_type'")
            
    # Initialise a model class
    try:
        model = LanguageModel(
            vocab_size=tokenizer_vocab_size,
            block_size=model_params['block_size'],
            n_embedd=model_params['n_embedd'],
            attention_size=model_params['attention_size'],
            num_heads=model_params['num_heads'],
            num_layers=model_params['num_layers'],
            dropout=model_params['dropout'],
            positional_encoder_type=model_params['positional_encoder_type'],
            device=device
        )
        model = model.to(device=device)


    except Exception as e:
        raise RuntimeError("Failed to initialise a `LanguageModel` object") from e

    # Validate training params
    if not isinstance(training_params, dict):
        raise ValueError("Argument `training_params` must be of type dict.")

    elif not set(list(training_params.keys())).issuperset(set(['learning_rate', 'batch_size', 'steps'])):
        raise ValueError("Argument `training_params` must be of type dict and must have keys ['learning_rate', 'batch_size', 'steps']")
    
    # Validate the batch_size
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
        
    # Validate the batch_size
    elif not isinstance(training_params['steps'], int):
        try:
            training_params['steps'] = int(training_params['steps'])

        except Exception as e:
            raise TypeError("Argument `training_params['steps']` must be of type `int`") from e  
        
    else:
        print("Starting Training")

    # Initialise an optimizer object
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=training_params['learning_rate'])

    except Exception as e:
        raise RuntimeError("Failed to initialise a `Optimizer` object") from e
    
    # Change check_point_params
    check_point_params['save_dir'] = f"{runs_dir}/run-{run_number}/checkpoints"
    
    # Parsing the model and optimizer to the training theory
    model, losses = lazy_batch_trainer(dir_path=array_directory,
                                       model=model,
                                       optimizer=optimizer,
                                       batch_size=training_params['batch_size'],
                                       block_size=model_params['block_size'],
                                       steps=training_params['steps'],
                                       device=device,
                                       check_point_params=check_point_params,
                                       train_ratio=0.95)
    
    # Save loss to a directory
    if save_loss_curves:
        plot_loss(losses, f"{runs_dir}/run-{run_number}/loss-logs", smoothen=smoothen_loss_plots)

    # Save model, optimizer and params
        # Model Params
    model_params = {
        "block_size": model_params['block_size'],
        "batch_size": training_params['batch_size'],
        "split_ratio": 0.9,
        "steps": training_params['steps'],
        "max_tokens": model_params['block_size'],
        "learning_rate": training_params['learning_rate'],
        "n_embedd": model_params['n_embedd'],
        "attention_size": model_params['attention_size'],
        "dropout": model_params['dropout'],
        "num_layers": model_params['num_layers'],
        "num_heads": model_params['num_heads'],
        "tokenizer_type" : "tiktoken",
        "tokenizer_encoding" : tokenizer_encoding,
        "vocab_size" : tokenizer_vocab_size,
        "positional_encoder_type" : model_params['positional_encoder_type']
    }
    torch.save(model.state_dict(), f"{runs_dir}/run-{run_number}/model/LanguageModel.pt")
    torch.save(optimizer.state_dict(), f"{runs_dir}/run-{run_number}/model/Optimizer.pt")
    with open (f"{runs_dir}/run-{run_number}/model/params.json", "w") as f:
        json.dump(model_params, f)

    # Save the losses to the npy file
    losses = np.array(losses)
    np.save(f"{runs_dir}/run-{run_number}/loss-logs/losses.npy", losses)

    return model, losses



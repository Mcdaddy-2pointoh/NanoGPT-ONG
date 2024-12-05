# Imports
from data_processing.loaders import text_data_loader
from data_processing.augmenters import  train_test_splitter, batch_generator, get_vocab
from tokenizers.naive_tokenizer import naive_tokenizer
from tokenizers.tiktokenizer import tiktokenizer
from model.models import LanguageModel
from trainers.naive.trainer import naive_trainer
import torch
from telemetry.visualisers import plot_loss
import os
import numpy as np
import json

# Training Pipeline
def training_pipeline(dir_path, block_size, batch_size, split_ratio, steps, max_tokens=300, save_loss_curves: bool = True, learning_rate: float = 1e-3, n_embedd: int = 32, attention_size: int = 32, dropout: float = 0.20, num_layers: int = 6, num_heads: int = 4, tokenizer_type: str = "tiktoken", tokenizer_encoding: str = "", runs_dir: str = "./runs", smoothen_loss_plots: bool = False, positional_encoder_type: str= "sinusoidal"):
    """
    Function: training_pipeline pipeline to train 
    Args:
        dir_path (str): Path to the directory containing the text files
        block_size (int): Block size is the maximum context window of the model
        batch_size (int): Batch size is the number of concurrent samples we can load for GPU saturation
        split_ratio (1> float > 0): The ratio in which the data must be split for training and testing
        steps (int) : The number of steps to train the model for
        max_tokens(int): The max number of new tokens to generate from 
        save_loss_curves(bool): Saves the loss curve as a png in the directory (./runs/run_number/loss-logs)
        learning_rate(float): Learning rate fed to the optimizer
        n_embedd (int): Linear dimension in which the token in projected into
        attention_size (int): The projection dimension of all attention heads combined
        dropout (1> float >0): The dropout ratio 
        num_layers (int): The number of replicated decoder only blocks
        num_heads (int): Number of attention heads to parallelize the attention mechanism
        tokenizer_type (ENUM(str)): The type of tokenizer used
        tokenizer_encoding (str):  `Empty string` for naive tokenizer but needed for tiktokenizer
        runs_dir (str): Path to the directory storing telemetry of each training run
        smoothen_loss_plots (bool): Smoothen the loss curve while plotting the train loss
        positional_encoder_type (ENUM(str)): The type of positional encoder used
    """

    # vaidate if the run directory exists else make one & create a run number 
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

    # Check device 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data using text_data_loader
    try:
        plain_text_data = text_data_loader(dir=dir_path)
    except Exception as e:
        raise RuntimeError("Could not load data from `dir_path`") from e

    print("Loaded Data Successfully!")

    # Initialise a tokenizer & tokenized data
    if tokenizer_type not in ["naive", "tiktoken"]:
            raise ValueError(f"Argument tokenizer_type must be 'naive' or 'tiktoken', found {tokenizer_type}")

    # Instancing a `naive tokenizer`
    elif tokenizer_type == "naive":
        
        # Get vocab details
        try: 
            vocab_characters, vocab_size = get_vocab(plain_text_data)
        except Exception as e:
            raise RuntimeError("Could not generate vocabulary from plain_text_data") from e
        
        # instance a tokenizer
        try:
            tokenizer = naive_tokenizer(vocab_characters=vocab_characters) 
            tokenized_data = tokenizer.encode(string=plain_text_data) 
        except Exception as e:
            raise RuntimeError("Could not tokenize string") from e

    # Instancing a tiktokenizer 
    elif tokenizer_type == "tiktoken":
        try:
            tokenizer = tiktokenizer(encoding=tokenizer_encoding)
            tokenized_data = tokenizer.encode(string=plain_text_data)
            vocab_size = max(tokenized_data) + 1

        except Exception as e:
            raise RuntimeError("Could not tokenize string") from e
        
    # Else Tokenization failed raise error
    else:
        raise RuntimeError("Tokenization failed")

    print("Tokenized Data Successfully!")
    
    
    # Setting positional encoders
    if not isinstance(positional_encoder_type, str):
        raise TypeError("Argument `positional_encoder_type` must be of type str")
    
    # Set `positional_encoder_type` to naive
    elif positional_encoder_type == "naive":
        positional_encoder_type = "naive"
        
    # Set `positional_encoder_type` to sinusoidal
    elif positional_encoder_type == "sinusoidal":
        positional_encoder_type = "sinusoidal"

    # Else key `positional_encoder_type` is out of bounds raise error
    else:
        raise ValueError("Argument `positional_encoder_type` must be either 'RoPE', 'sinusoidal' or 'naive'")

    
    # Split the data into train test split & Create input and output tensors
    try: 
        train_split, test_split = train_test_splitter(data=tokenized_data, split_ratio=split_ratio) 
        
    except Exception as e:
        raise RuntimeError("Could not transform tokenized data") from e
    
    # Make a  model & test a sample result
    x_batch, y_batch = batch_generator(data=train_split, block_size=block_size, batch_size=batch_size, as_torch_tensors=True, device=device)
    
    print("Training Begins")
    
    model = LanguageModel(vocab_size=vocab_size, 
                                block_size=block_size, 
                                n_embedd=n_embedd,
                                device=device,
                                attention_size=attention_size,
                                num_heads= num_heads,
                                num_layers=num_layers,
                                dropout = dropout,
                                positional_encoder_type=positional_encoder_type
                                )
    model = model.to(device=device)
    logits, loss = model(x_batch, y_batch)

    # Instance an optimizer 
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Train the model 
    model, losses = naive_trainer(data=tokenized_data, model=model, optimizer=optimizer, batch_size=batch_size, block_size=block_size, steps=steps)

    # Predict from the model
    idx = torch.zeros((1,1), dtype=torch.long).to(device=device)
    preds = tokenizer.decode(model.generate(idx=idx, max_new_tokens=max_tokens)[0].tolist())

    # Save predictions to file
    with open(f"{runs_dir}/run-{run_number}/results/preds.txt", "x", encoding="utf-8") as file:
        file.write(preds)

    # Save loss to a directory
    if save_loss_curves:
        plot_loss(losses, f"{runs_dir}/run-{run_number}/loss-logs", smoothen=smoothen_loss_plots)

    # Save model, optimizer and params
        # Model Params
    model_params = {
        "block_size": block_size,
        "batch_size": batch_size,
        "split_ratio": split_ratio,
        "steps": steps,
        "max_tokens": max_tokens,
        "learning_rate": learning_rate,
        "n_embedd": n_embedd,
        "attention_size": attention_size,
        "dropout": dropout,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "tokenizer_type" : tokenizer_type,
        "tokenizer_encoding" : tokenizer_encoding,
        "vocab_size" : vocab_size,
        "positional_encoder_type" : positional_encoder_type
    }
    torch.save(model.state_dict(), f"{runs_dir}/run-{run_number}/model/LanguageModel.pt")
    torch.save(optimizer.state_dict(), f"{runs_dir}/run-{run_number}/model/Optimizer.pt")
    with open (f"{runs_dir}/run-{run_number}/model/params.json", "w") as f:
        json.dump(model_params, f)

    # Save the losses to the npy file
    losses = np.array(losses)
    np.save(f"{runs_dir}/run-{run_number}/loss-logs/losses.npy", losses)

    # Get encoding and decoding hashmaps of tokenizer
    if tokenizer_type == "naive":
        tokenizer_encoding_hashmap = tokenizer.get_encoder_hashmap()
        tokenizer_decoding_hashmap = tokenizer.get_decoder_hashmap()
        tokenizer_characters = tokenizer.get_characters()

        # Save tokeinzer params
        with open (f"{runs_dir}/run-{run_number}/tokenizer/encoding_hashmap.json", "w") as f:
            json.dump(tokenizer_encoding_hashmap, f)

        with open (f"{runs_dir}/run-{run_number}/tokenizer/decoding_hashmap.json", "w") as f:
            json.dump(tokenizer_decoding_hashmap, f)

        with open (f"{runs_dir}/run-{run_number}/tokenizer/characters.json", "w") as f:
            json.dump(tokenizer_characters, f)

    return {"model": model, "losses": losses, "preds": preds}
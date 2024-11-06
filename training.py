# Imports
from utils.data.loaders import text_data_loader
from utils.data.augmenters import naive_tokenizer, train_test_splitter, batch_generator, get_vocab
from utils.modelling.models import LanguageModel
from utils.modelling.trainers import naive_trainer
import torch
from utils.telemetry.visualisers import plot_loss
import os
import numpy as np
import json

# Main pipeline
def main(dir_path, block_size, batch_size, split_ratio, steps, max_tokens=300, save_loss_curves: bool = True, learning_rate: float = 1e-3, n_embedd: int = 32, attention_head_size: int = 32, dropout: float = 0.20, num_layers: int = 6, num_heads: int = 4):
    """
    Function: Main pipeline to train 
    Args:
        dir_path (str): Path to the directory containing the text files
        block_size (int): Block size is the maximum context window of the model
        batch_size (int): Batch size is the number of concurrent samples we can load for GPU saturation
        split_ratio (1> float > 0): The ratio in which the data must be split for training and testing 
        max_tokens(int): The max number of new tokens to generate from 
        save_loss_curves(bool): Saves the loss curve as a png in the directory (./runs)
        learning_rate(float): Learning rate fed to the optimizer
        n_embedd(int): Embedding Dimensions for 
    """

    # Create run number
    run_number = str(len(os.listdir("./runs")) + 1).zfill(4)

    # Create a run directory
    os.mkdir(f"./runs/run-{run_number}")
    os.mkdir(f"./runs/run-{run_number}/loss-logs")
    os.mkdir(f"./runs/run-{run_number}/results")
    os.mkdir(f"./runs/run-{run_number}/model")
    os.mkdir(f"./runs/run-{run_number}/tokenizer")

    # Check device 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data using text_data_loader
    try:
        plain_text_data = text_data_loader(dir=dir_path)
    except Exception as e:
        raise RuntimeError("Could not load data from `dir_path`") from e

    # Get vocab details
    try: 
        vocab_characters, vocab_size = get_vocab(plain_text_data)
    except Exception as e:
        raise RuntimeError("Could not generate vocabulary from plain_text_data") from e

    # Initialise a tokenizer & tokenized data
    try:
        tokenizer = naive_tokenizer(vocab_characters=vocab_characters) 
        tokenized_data = tokenizer.encode(string=plain_text_data) 
    except Exception as e:
        raise RuntimeError("Could not tokenize string") from e


    # Split the data into train test split & Create input and output tensors
    try: 
        train_split, test_split = train_test_splitter(data=tokenized_data, split_ratio=split_ratio) 
        
    except Exception as e:
        raise RuntimeError("Could not transform tokenized data") from e
    
    # Make a  model & test a sample result
    x_batch, y_batch = batch_generator(data=train_split, block_size=block_size, batch_size=batch_size, as_torch_tensors=True, device=device)
    model = LanguageModel(vocab_size=vocab_size, 
                                block_size=block_size, 
                                n_embedd=n_embedd,
                                device=device,
                                attention_head_size=attention_head_size,
                                num_heads= num_heads,
                                num_layers=num_layers,
                                dropout = dropout
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
    with open(f"./runs/run-{run_number}/results/preds.txt", "x", encoding="utf-8") as file:
        file.write(preds)

    # Save loss to a directory
    if save_loss_curves:
        plot_loss(losses, f"./runs/run-{run_number}/loss-logs", smoothen=False)

    # Save model
    torch.save(model.state_dict(), f"./runs/run-{run_number}/model/BigramModel.pt")
    torch.save(optimizer.state_dict(), f"./runs/run-{run_number}/model/Optimizer.pt")

    # Save the losses to the npy file
    losses = np.array(losses)
    np.save(f"./runs/run-{run_number}/loss-logs/losses.npy", losses)

    # Get encoding and decoding hashmaps of tokenizer
    tokenizer_encoding_hashmap = tokenizer.get_encoder_hashmap()
    tokenizer_decoding_hashmap = tokenizer.get_decoder_hashmap()
    tokenizer_characters = tokenizer.get_characters()

    # Save tokeinzer params
    with open (f"./runs/run-{run_number}/tokenizer/encoding_hashmap.json", "w") as f:
        json.dump(tokenizer_encoding_hashmap, f)

    with open (f"./runs/run-{run_number}/tokenizer/decoding_hashmap.json", "w") as f:
        json.dump(tokenizer_decoding_hashmap, f)

    with open (f"./runs/run-{run_number}/tokenizer/characters.json", "w") as f:
        json.dump(tokenizer_characters, f)

    return {"model": model, "losses": losses, "preds": preds}

results = main(dir_path="./data/",
              block_size=1024,
              batch_size=12,
              steps=2500,
              split_ratio= 0.8,
              save_loss_curves=True,
              learning_rate=3e-4,
              max_tokens=1024,
              n_embedd = 128,
              num_heads= 4,
              attention_head_size = 128,
              num_layers= 6,
              dropout= 0.20
              )

# Visualise
print(results["preds"])




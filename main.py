# Imports
from utils.loaders import text_data_loader
from utils.augmenters import naive_tokenizer, train_test_splitter, batch_generator, get_vocab
from utils.model import BigramLanguageModel
import torch
import tqdm

# Main pipeline
def main(dir_path, block_size, batch_size, split_ratio, max_tokens=100):
    """
    Function: Main pipeline to train 
    Args:
        dir_path (str): Path to the directory containing the text files
        block_size (int): Block size is the maximum context window of the model
        batch_size (int): Batch size is the number of concurrent samples we can load for GPU saturation
        split_ratio (1> float > 0): The ratio in which the data must be split for training and testing 
        max_tokens(int): The max number of new tokens to generate from bigram
    """

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
    
    # Make a bigram model & test a sample result
    x_batch, y_batch = batch_generator(data=train_split, block_size=block_size, batch_size=batch_size)
    model = BigramLanguageModel(vocab_size=vocab_size)
    logits, loss = model(x_batch, y_batch)

    # Make a prediction
    idx = torch.zeros((1,1), dtype=torch.long)
    preds = tokenizer.decode(model.generate(idx=idx, max_new_tokens=max_tokens)[0].tolist())

    # Instance an optimizer 
    # optimizer = torch.optim.AdamW(model.parameters, lr=1e-3)

    # Train the model 
    

    return preds

result = main(dir_path="./data/",
              block_size=8,
              batch_size=4,
              split_ratio= 0.8,
              )

print(result)
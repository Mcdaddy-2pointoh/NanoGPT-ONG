import os
import torch
import math as m
import numpy as np

# Create a vocabulary
def get_vocab(text: str) -> tuple:
    """
    Function: Converts a corpoora of text to vocabulary of tokens and vocab size
    Args:
        text (str): Overall corpora as a string
    """
    # ValueError
    if type(text) != str:
        raise TypeError("Argument `txt` must be of type string")
    
    else: 
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        return chars, vocab_size
    
# Create a tokenizer elements 
class naive_tokenizer():
    def __init__(self, vocab_characters):
        # Raise error if vocab_characters is not of type list
        if type(vocab_characters) != list:
            raise TypeError("Argument `vocab_characters` must of type list.")

        # Mapping object propertiees
        self.characters = vocab_characters
        self.encoder_hash_map = {character: index for index, character in enumerate(vocab_characters)}
        self.decoder_hash_map = {index: character for index, character in enumerate(vocab_characters)}

    def encode(self, string: str)-> list:
        """
        Function: Method to encode a str to a token
        Arg:
            string (str): String character to encode to token list
        """
        if string == "" or type(string) != str:
            return None
        else:
            return [self.encoder_hash_map[character] for character in string]
    
    def decode(self, tokens: list)-> str:
        """
        Function: Method to decode a token list to a string
        Arg:
            tokens (list): Loist of tokens to decode to plaintext
        """        
        if tokens == [] or type(tokens) != list:
            return None
        
        else:
            return "".join([self.decoder_hash_map[token] for token in tokens])
        
# Train validation splitter
def train_test_splitter(data: list, split_ratio: float)-> tuple:
    """
    Function: Splits a list of data into train and test split
    Args:
        data (list): Data to be split into train and test split
        split_ratio (float): The ratio in which the data must be split for training and testing 
    """

    # Type check input
    if type(split_ratio) != float:
        try:
            split_ratio = float(split_ratio)
        except ValueError as e:
            raise TypeError("The argument `split_ratio` must be a floating point number in the range of 0-1") from e

    elif split_ratio > 1 or split_ratio < 0:
        raise TypeError("The argument `split_ratio` must be a floating point number in the range of 0-1")

    elif type(data) != list or data == []:
        raise TypeError("The argument `data` must be a non empty list")

    # Return the train test split
    else:
        train_test_boundary = m.floor(len(data) * split_ratio)
        return data[:train_test_boundary], data[train_test_boundary:]

# Input target generator
def batch_generator(data: list, block_size:int, batch_size: int)-> tuple:
    """
    Funtion: Produces `batch_size` input and target batches, with each input & target a `block_size` len tensor
    Args:
        data (list): The dataset of tokens
        block_size (int): Context length of the model. Time dimesion
        batch_size (int): The concurrent training samples a model can take to keep the GPUs utillized
    """

    # Validate inputs
    if type(data) != list:
        try:
            data = [int(i) for i in list(data)]
        except ValueError as e:
            raise ValueError("The argument `data` must be a non empty list with integers")

    elif type(block_size) != int:
        try:
            block_size = int(block_size)
        except ValueError as e:
            raise ValueError("The argument `block_size` must be of type int")

    elif type(batch_size) != int:
        try:
            batch_size = int(batch_size)
        except ValueError as e:
            raise ValueError("The argument `batch_size` must be of type int")

    # Return batch and target pairs
    else:
        indices = torch.randint(0, len(data) - block_size, (batch_size,))
        x = [data[index:index+block_size] for index in indices]
        y = [data[index+1:index+block_size+1] for index in indices]

        return x, y
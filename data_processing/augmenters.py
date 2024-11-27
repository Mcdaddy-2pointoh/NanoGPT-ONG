import torch
import math as m
import os
from tqdm import tqdm
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
            raise TypeError("The argument `split_ratio` must be a floating point number in the range of 0-1") from e.with_traceback()

    elif split_ratio > 1 or split_ratio < 0:
        raise TypeError("The argument `split_ratio` must be a floating point number in the range of 0-1")

    elif type(data) != list or data == []:
        raise TypeError("The argument `data` must be a non empty list") 

    # Return the train test split
    else:
        train_test_boundary = m.floor(len(data) * split_ratio)
        return data[:train_test_boundary], data[train_test_boundary:]

# Input target generator
def batch_generator(data: list, block_size:int, batch_size: int, as_torch_tensors=True, device: str = "cpu")-> tuple:
    """
    Funtion: Produces `batch_size` input and target batches, with each input & target a `block_size` len tensor
    Args:
        data (list): The dataset of tokens
        block_size (int): Context length of the model. Time dimesion
        batch_size (int): The concurrent training samples a model can take to keep the GPUs utillized
        as_torch_tensors (bool): Returns xb, yb as torch.Tensor objects
    """

    # Validate inputs
    if type(data) != list and type(data) != torch.Tensor:
        try:
            data = [int(i) for i in list(data)]
        except ValueError as e:
            raise ValueError("The argument `data` must be a non empty list with integers") from  e.with_traceback()

    elif type(block_size) != int:
        try:
            block_size = int(block_size)
        except ValueError as e:
            raise ValueError("The argument `block_size` must be of type int") from e.with_traceback()

    elif type(batch_size) != int:
        try:
            batch_size = int(batch_size)
        except ValueError as e:
            raise ValueError("The argument `batch_size` must be of type int") from  e.with_traceback()

    # Return batch and target pairs
    else:
        indices = torch.randint(0, len(data) - block_size, (batch_size,))
        x = [data[index:index+block_size] for index in indices]
        y = [data[index+1:index+block_size+1] for index in indices]

        # Convert dtype to torch.Tensor
        if as_torch_tensors:
            x = torch.stack([torch.tensor(data[index:index+block_size]) for index in indices])
            y = torch.stack([torch.tensor(data[index+1:index+block_size+1]) for index in indices])
            return x.to(device=device), y.to(device=device)
        
        else:
            return x, y
        
# Split files on lines
def file_splitter(data: str, target_dir: str, split_threshold: int, write_frequency: int = 5000, file_encoding: str = "utf-8", verbose: bool = False):
    """
    Function: Splits file into smaller segment file containing `split_threshold` number of lines
    Args:
        data (str): Path of the file to be split
        target_dir (str): Path to dump segmented files
        write_frequency (int): Writes into the segment file after write_frequency lines are read
        split_threshold (int): Number of lines to split the data into
        file_encoding (str): File encoding format to read and write the data file, conventionally `utf-8` 
        verbose (bool): Prints the line operated upon 
    """

    # Validate data
    if not isinstance(data, str):
        raise TypeError("Argument `data` must be  of type str, pointing to a .txt file")
    
    elif not os.path.isfile(data):
        raise FileNotFoundError(f"Path `{data}` doesn't exist or is not a file")
    
    elif data.split(".")[-1] != "txt":
        raise TypeError(f"Path `{data}` must be of type .txt")
    
    elif not isinstance(file_encoding, str):
        raise TypeError(f"Argument `file_encoding` must be of type string and a valid file encoding format")
    
    # Validate target_dir
    elif not os.path.isdir(target_dir):
        raise FileNotFoundError(f"Argument `target_dir` must be the path to an existing file.")
    
    # Split threshold is not an int
    elif not isinstance(split_threshold, int):
        raise TypeError(f"Argument `split_threshold` must be of type int.")
    
    # Else raise a value
    else:
        segment_number = len(os.listdir(target_dir))
        segment_number = str(segment_number).zfill(7)
        string = ""

        # Read file
        try:
            with open(data, 'r', encoding=file_encoding) as txt_file:

                # for loop to read a line at a time
                for index, line in enumerate(txt_file):

                    # If verbose print the iiteration
                    if verbose:
                        print(f"Line Number: {index + 1}")

                    # Change the segment number
                    if index % split_threshold == 0:
                        
                        # Change the file number
                        segment_number = int(segment_number) + 1
                        segment_number = str(segment_number).zfill(7)

                    # Write the string to the temporary variable 
                    string += f"{line}"
                    
                    # Write to segment and save
                    if index % write_frequency == 0:
                        segment_file_path = os.path.join(target_dir, f"segment-{segment_number}.txt")
                        
                        # Save segmented file 
                        try: 
                            with open(segment_file_path, "a+", encoding=file_encoding) as segment:
                                segment.write(string)
                                
                            string = ""
                        
                        except Exception as e:
                            raise RuntimeError(f"Could not save the segment") from e

            return target_dir
        
        except Exception as e:
            raise RuntimeError(f"Could not segment the file") from e
            

# Function to convert file to numpy array with tokenization 
def segmented_tokenization(data_dir: str,
                  tokenizer,
                  file_encoding: str,
                  target_dir: str):
    
    """
    Function: Takes in the directory of segment files and outputs the tokenized list of the each files and .npy files
    Args:
        data_dir (str): The path to the directory containing the text segments4
        tokenizer: Tik token object to be used for text tokenization
        file_encoding (str): Encoding format of the text file
        target_dir (str): Path to the directory to save the encoded npy files
    """
    
    for file in tqdm(os.listdir(data_dir)):
        
        # Get file metadata
        file_path = os.path.join(data_dir, file) if os.path.isfile(os.path.join(data_dir, file)) else "dir.dir"

        # Skip dirs
        if file_path == "dir.dir":
            continue

        # Else extract file information
        file_name, file_extension = file.split(".")
        save_path = os.path.join(target_dir, file_name)

        # If extension is not `.txt` pass
        if file_extension != "txt":
            continue

        # Open a file for tokenization
        with open(file_path, "r", encoding=file_encoding) as read_file:
            string = read_file.read()

        # Tokenize the string and store
        tokenized_data = tokenizer.encode(string=string)

        # Convert the data to a numpy array and save to file
        np_tokenized_data = np.array(tokenized_data)

        # Cleaning memory prior to rw
        del tokenized_data 
        del read_file

        # Saving npy
        np.save(save_path, np_tokenized_data)
    
    # Return 
    return target_dir
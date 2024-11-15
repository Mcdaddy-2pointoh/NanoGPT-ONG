from utils.data.augmenters import file_splitter
import os
import torch
import shutil
from tqdm import tqdm
from utils.tokenizers.tiktokenizer import tiktokenizer

def lazy_batch_training(
    data: str,
    file_splitter_params: dict,
    tokenizer_encoding: str,
    segment_data: bool = True,
    runs_dir: str = "./runs"
):
    
    # SETTING UP RUNS
    # Create a directory to store all runs metadata
    # Validate compute devices available on the PC
    # validate if the run directory exists else make one & create a run number 
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
            if not set(file_splitter_params.keys()).issuperset(set(['segment_target_dir', 'array_target_dir', 'split_threshold', 'verbose', 'file_encoding', 'write_frequency'])):
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
                value = str(input("File dir is not empty do you want to overwrite existing file `Y\N`?")).upper()

                if value not in ['Y', 'N']:
                    raise ValueError("Invalid value typed in expected str 'Y' or 'N'")
                
                elif value == "Y":
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
                    
                
                else:
                    value = str(input("File dir is not empty do you want to partition data again without overwriting `Y\N`?")).upper()

                    # Partition data without overwriting
                    if value == "Y":
                        print("Please monitor CPU stats while splitting files")
                        file_splitter(
                                data=data,
                                target_dir=file_splitter_params['segment_target_dir'],
                                split_threshold=file_splitter_params['split_threshold'], # 200k lines per file segment
                                write_frequency=file_splitter_params['write_frequency'],
                                file_encoding=file_splitter_params['file_encoding'],
                                verbose=file_splitter_params['verbose'])  
                        
                    # Skipping partioning
                    elif value == "N":
                        data_dir = file_splitter_params['segment_target_dir']
                        print("Skipped Partitioning")

                    # Raise error for incorrect input
                    else:
                        raise ValueError("Invalid value typed in expected str 'Y' or 'N'")

            # If file is empty just begin partitioning
            else:
                print("Please monitor CPU stats while splitting files")
                data_dir = file_splitter(
                        data=data,
                        target_dir=file_splitter_params['segment_target_dir'],
                        split_threshold=file_splitter_params['split_threshold'], # 200k lines per file segment
                        write_frequency=file_splitter_params['write_frequency'],
                        file_encoding=file_splitter_params['file_encoding'],
                        verbose=file_splitter_params['verbose'])                  
        
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
            os.makedirs(file_splitter_params['array_target_dir'])

        except Exception as e:
            raise RuntimeError("Could not create directories") from e

    # Else if the directory has some arrays already present 
    elif os.path.isdir(file_splitter_params['array_target_dir']) and :



    for file in tqdm(os.listdir(data_dir)):
        
        # Get file metadata
        file_path = os.path.join(data_dir, file)
        file_name, file_extension = file.split(".")

        # Open a file for tokenization
        with open(file_path, "r", file_splitter_params['file_encoding']) as read_file:
            string = read_file.read()

        # Tokenize the string and store
        tokenized_data = tokenizer.encode(string=string)

        # Convert the data to a numpy array and save to file

        

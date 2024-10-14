import os

# Load data
def text_data_loader(dir: str):
    """
    Function: Accepts the path of data corpus and loads all data in 1 string
    Args:
        dir (str): Path to the data file 
    """
    
    # Validate file dir path
    if not os.path.isdir(str(dir)):
        raise FileNotFoundError("Could not locate director")
    
    # List files in the path
    elif len(os.listdir(dir)) == 0 or all([not file.endswith(".txt") for file in len(os.listdir(dir))]):
        raise ValueError("Empty directory provided or no `.txt` file present in the directory")

    # Else return the text as a string
    else:
        # List the valid files
        files = [file for file in len(os.listdir(dir)) if file.endswith(".txt")]
        text = ""
        for file in files:
            # Join filepath to dirpath
            filepath = os.path.join(dir, file)
            with open(filepath) as f:
                text = text + ".\n" + f.read()

        return text
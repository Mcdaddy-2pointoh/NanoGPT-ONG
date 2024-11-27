import os

# Load data
def text_data_loader(dir: str, encoding: str = "utf8"):
    """
    Function: Accepts the path of data corpus and loads all data in 1 string
    Args:
        dir (str): Path to the data file 
        encoding (str) : File encoding format if any, default is utf8
    """
    
    # Validate file dir path
    if not os.path.isdir(str(dir)):
        raise FileNotFoundError("Could not locate director")
    
    # List files in the path
    elif len(os.listdir(dir)) == 0:
        raise ValueError("Empty directory provided or no `.txt` file present in the directory")

    # Else return the text as a string
    else:
        # List the valid files
        files = [file for file in os.listdir(dir) if file.endswith(".txt")]
        if len(files) == 0: 
            raise ValueError("No file with .txt extension found")
        text = ""
        for file in files:
            # Join filepath to dirpath
            filepath = os.path.join(dir, file)

            try:
                with open(filepath, encoding=encoding) as f:
                    file_text = f.read()
                    text = text + "\n" + file_text
            except Exception as e:
                raise RuntimeError("Error Reading the file check for the encoding format.")

        return text
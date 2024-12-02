import tiktoken

class tiktokenizer:
    """
    Class: Tokenizer class that aids in producing BPE based tokenization of text
    """

    def __init__(self, encoding: str = "cl100k_base"):
        """
        Function: Initialises an object of class `tiktokenizer`
        Args:
            encoding (Enum(str)): The type of encoding used by the language model for tokenization
        """

        #  Type check the encoding
        if not isinstance(encoding, str):
            raise TypeError("Argument encoding must be of type `Enum(str)`")
        
        # Try to load the tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding(encoding)

        except Exception as e:
            raise RuntimeError(f"Could not load tokenizer with encoding {encoding} from tiktokens") from e
        
    def encode(self, string: str):
        """
        Function: Method to encode a str to a token
        Arg:
            string (str): String character to encode to token list
        """

        # Valdiate string input
        if string == "" or type(string) != str:
            raise ValueError("Argument `tokens` must be a non empty value of type list")

        # Pass to the encoder
        else:
            tokens = self.tokenizer.encode(text=string)
            return tokens
    

    def decode(self, tokens: list):
        """
        Function: Method to decode a token list to a string
        Arg:
            tokens (list): List of tokens to decode to plaintext
        """  

        # Validate token input
        if tokens == [] or type(tokens) != list:
            raise ValueError("Argument `tokens` must be a non empty value of type list")
        
        # Pass the tokens to decoder
        else:
            string = self.tokenizer.decode(tokens=tokens)
            return string


        
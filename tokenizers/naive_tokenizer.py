# Create a tokenizer elements 
class naive_tokenizer():
    def __init__(self, vocab_characters: list = [], encoder_hash_map: dict = {}, decoder_hash_map: dict = {}, from_hashmaps: bool = False):
        # if loading tokenizer from hashmaps
        if from_hashmaps:

            # For invalid dtype encoder hashmaps
            if type(encoder_hash_map) != dict:
                raise ValueError("Argument `encoder_hash_map` must be of type dict")
            
            # For empty encoder hash maps
            elif encoder_hash_map == {}:
                raise ValueError("The arg `encoder_hash_map` cannot be empty when creating tokenizer from hashmap")
            
            # For invalid dtype encoder hashmaps
            elif type(decoder_hash_map) != dict:
                raise ValueError("Argument `decoder_hash_map` must be of type dict")
            
            # For empty encoder hash maps
            elif decoder_hash_map == {}:
                raise ValueError("The arg `decoder_hash_map` cannot be empty when creating tokenizer from hashmap")
            
            # Validate type and set `encoder_hash_map` & `decoder_hash_map`
            encoder_hash_map = {str(k): int(v) for k,v in encoder_hash_map.items()}
            decoder_hash_map = {int(k): str(v) for k,v in decoder_hash_map.items()}
            
            # All keys of encoder must be a value in decoder
            k_encoder = set(encoder_hash_map.keys())
            v_decoder = set(decoder_hash_map.values())
            if (k_encoder != v_decoder): 
                raise RuntimeError("Key Value mismatch in the encoder-decoder hashmaps")


            # All keys of encoder must be a value in decoder
            v_encoder = set(encoder_hash_map.values())
            k_decoder = set(decoder_hash_map.keys())
            if (v_encoder != k_decoder): 
                raise RuntimeError("Key Value mismatch in the encoder-decoder hashmaps")

            self.encoder_hash_map = encoder_hash_map
            self.decoder_hash_map = decoder_hash_map
            self.characters = k_encoder

        # If initialising a tokenizer from characters 
        else:

            # Raise error if vocab_characters is not of type list
            if type(vocab_characters) != list or len(vocab_characters) == 0:
                raise TypeError("Argument `vocab_characters` must of type list and must not be an empty list.")

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
            raise ValueError("Argument `string` must be a non empty value of type string")
        else:
            return [self.encoder_hash_map[character] for character in string]
    
    def decode(self, tokens: list)-> str:
        """
        Function: Method to decode a token list to a string
        Arg:
            tokens (list): List of tokens to decode to plaintext
        """        
        if tokens == [] or type(tokens) != list:
            raise ValueError("Argument `tokens` must be a non empty value of type list")
        
        else:
            return "".join([self.decoder_hash_map[token] for token in tokens])
        
    def get_encoder_hashmap(self):
        """
        Function to return the encoder hashmap
        """
        return self.encoder_hash_map
    

    def get_decoder_hashmap(self):
        """
        Function to return the decoder hashmap
        """
        return self.decoder_hash_map
    
    def get_characters(self):
        """
        Function to return the vocab characters
        """
        return self.characters

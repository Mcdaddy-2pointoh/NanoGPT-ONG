# Imports
from tokenizers.naive_tokenizer import naive_tokenizer
from tokenizers.tiktokenizer import tiktokenizer
from model.models import LanguageModel
import torch
import json
import os

class InferencePipeline:
    """
    Class: Inference pipeline to run the language model 
    """

    def __init__(self, run_dir: str = None, device: str = "cpu"):
        """
        Function: Accepts an input string and produces an output from Language model
        Args:
            run_dir (str): Model run directory that stores model and tokenizer information from the run
            tokenizer_type (Enum): Tokenizer type
        """

        # Get a list of all possible devices
        devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())] + ['cpu']
        
        # Get model params as dictionary
        with open(os.path.join(run_dir, "model", "params.json")) as f:
            model_params = json.load(f)

        self.model_params = model_params

        # Try to instance the tokenizer
        if model_params['tokenizer_type'] not in ["naive", "tiktoken"]:
            raise ValueError(f"Value model_params['tokenizer_type'] must be 'naive' or 'tiktoken', found {model_params['tokenizer_type']}")
        
        # Raise error if `naive` tokenizer is the type and the encoding and decoding hashmaps are not present
        elif model_params['tokenizer_type'] == "naive" and not set(os.listdir(os.path.join(run_dir, "tokenizer"))).issuperset(set(['characters.json', 'decoding_hashmap.json', 'encoding_hashmap.json'])):
            raise FileNotFoundError("To use naive tokenizer 'characters.json', 'decoding_hashmap.json'& 'encoding_hashmap.json' are needed for initialising the tokenizer")
        
        # Load hashmaps of encoding and decoding to initialise a transformer
        elif model_params['tokenizer_type'] == "naive":

            # Load hashmaps
            with open(os.path.join(run_dir, "tokenizer", 'encoding_hashmap.json')) as f:
                encoder_hash_map = json.load(f)

            with open(os.path.join(run_dir, "tokenizer", 'decoding_hashmap.json')) as f:
                decoder_hash_map = json.load(f)

            with open(os.path.join(run_dir, "tokenizer", 'characters.json')) as f:
                characters = json.load(f)

            # Load a tokenizer
            try:
                self.tokenizer = naive_tokenizer(from_hashmaps=True, encoder_hash_map=encoder_hash_map, decoder_hash_map=decoder_hash_map)
            except Exception as e:
                raise RuntimeError("Could not load naive tokenizer, please check `encoding_hashmap.json`, `decoding_hashmap.json` &  `characters.json` provided.") from e

        # Use tiktokens
        elif model_params['tokenizer_type'] == "tiktoken":
            try: 
                self.tokenizer = tiktokenizer(encoding=model_params['tokenizer_encoding'])
            except Exception as e:
                raise RuntimeError("Could not load tiktokenizer, please check params `tokenizer_type` & `tokenizer_encoding` in `params.json`") from e

        # Else tokenizer could not be loaded
        else:
            raise RuntimeError("Could not load Tokenizer")

        # Check the device
        if device not in devices:
            raise RuntimeError(f"Processing engine not found please set device to one of the following options {devices}")
        
        # Set device
        self.device = device

        # Load model on specified device and get the model params
        if set(os.listdir(os.path.join(run_dir, 'model'))).issuperset(set(['BigramModel.pt', 'Optimizer.pt', 'params.json'])):
            model_path = os.path.join(run_dir, "model", "BigramModel.pt")
            self.model = LanguageModel(vocab_size=len(encoder_hash_map.keys()), 
                                    block_size=model_params['block_size'], 
                                    n_embedd=model_params['n_embedd'],
                                    device=device,
                                    attention_size=model_params['attention_size'],
                                    num_heads= model_params['num_heads'],
                                    num_layers=model_params['num_layers'],
                                    dropout = 0,
                                    positional_encoder_type=model_params["positional_encoder_type"]
                                    )
            self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
            self.model = self.model.to(device=device)
            self.model.eval()

        elif set(os.listdir(os.path.join(run_dir, 'model'))).issuperset(set(['LanguageModel.pt', 'Optimizer.pt', 'params.json'])):
            model_path = os.path.join(run_dir, "model", "LanguageModel.pt")
            self.model = LanguageModel(vocab_size=model_params['vocab_size'], 
                                    block_size=model_params['block_size'], 
                                    n_embedd=model_params['n_embedd'],
                                    device=device,
                                    attention_size=model_params['attention_size'],
                                    num_heads= model_params['num_heads'],
                                    num_layers=model_params['num_layers'],
                                    dropout = 0,
                                    positional_encoder_type=model_params["positional_encoder_type"]
                                    )
            self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
            self.model = self.model.to(device=device)
            self.model.eval()

        # Raise error
        else:
            raise FileNotFoundError("To use language model 'BigramModel.pt', 'Optimizer.pt' & 'params.json' are needed for initialising the pytorch model")

    def generate(self, text:str = "Hi dear model", max_tokens: int = 512):
        """
        Function: to generate text from the language model
        Args: 
            text (str): String to start the generation
        """
        
        # Tokenize the inputs
        tokenized_text = self.tokenizer.encode(string=text)

        # Restrict the tokenized inputs to context length
        if len(tokenized_text) >= self.model_params['block_size']:
            raise RuntimeError(f"The provided string is too long to generate output upon please reduce the size of input to {self.model_params['block_size']} tokens")

        # Augment the data from (T, C) to a (B, T, C) tensor, add a batch dimension
        else:

            # Validate max tokens 
            if max_tokens > self.model_params['max_tokens']:
                raise ValueError(f"Argument `max_tokens` must be an int between 1 and {self.model_params['max_tokens']}")

            # Convert a list to Torch tensor
            tokenized_text_tensor = torch.tensor(tokenized_text)

            # Reshape the layer to B, T, C
            tokenized_text_tensor_btc = torch.reshape(tokenized_text_tensor, shape=(1, *tokenized_text_tensor.shape)).to(device=self.device)

            # Parse input to the model 
            with torch.no_grad():
                model_gen = self.model.generate(tokenized_text_tensor_btc, max_new_tokens=self.model_params['max_tokens'])

            # Squeezing model output
            model_gen = model_gen[0].tolist()
            output = self.tokenizer.decode(model_gen)

            # Clear local variables
            del(tokenized_text)
            del(tokenized_text_tensor)
            del(tokenized_text_tensor_btc)
            del(model_gen)

            return output

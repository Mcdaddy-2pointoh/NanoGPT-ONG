# Imports
from utils.tokenizers.naive_tokenizer import naive_tokenizer
from utils.modelling.models import LanguageModel
import torch
import json
from utils.telemetry.visualisers import plot_loss
import os
import numpy as np


class InferencePipeline:
    """
    Class: Inference pipeline to run the language model 
    """

    def __init__(self, run_dir: str = None, tokenizer_type: str = "naive", device: str = "cpu"):
        """
        Function: Accepts an input string and produces an output from Language model
        Args:
            run_dir (str): Model run directory that stores model and tokenizer information from the run
            tokenizer_type (Enum): Tokenizer type
        """

        # Get a list of all possible devices
        devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())] + ['cpu']

        # Try to instance the tokenizer
        if tokenizer_type not in ["naive", "tiktoken"]:
            raise ValueError(f"Argument tokenizer_type must be 'naive' or 'tiktoken', found {tokenizer_type}")
        
        # Raise error if `naive` tokenizer is the type and the encoding and decoding hashmaps are not present
        elif tokenizer_type == "naive" and not set(os.listdir(os.path.join(run_dir, "tokenizer"))).issuperset(set(['characters.json', 'decoding_hashmap.json', 'encoding_hashmap.json'])):
            raise FileNotFoundError("To use naive tokenizer 'characters.json', 'decoding_hashmap.json'& 'encoding_hashmap.json' are needed for initialising the tokenizer")
        
        # Load hashmaps of encoding and decoding to initialise a transformer
        elif tokenizer_type == "naive":

            # Load hashmaps
            with open(os.path.join(run_dir, "tokenizer", 'encoding_hashmap.json')) as f:
                encoder_hash_map = json.load(f)

            with open(os.path.join(run_dir, "tokenizer", 'decoding_hashmap.json')) as f:
                decoder_hash_map = json.load(f)

            with open(os.path.join(run_dir, "tokenizer", 'characters.json')) as f:
                characters = json.load(f)

            # Load a tokenizer
            self.tokenizer = naive_tokenizer(from_hashmaps=True, encoder_hash_map=encoder_hash_map, decoder_hash_map=decoder_hash_map)

        # Use tiktokens
        else:
            pass

        # Load the model Language model
        if not set(os.listdir(os.path.join(run_dir, 'model'))).issuperset(set(['BigramModel.pt', 'Optimizer.pt', 'params.json'])):
            raise FileNotFoundError("To use language model 'BigramModel.pt', 'Optimizer.pt' & 'params.json' are needed for initialising the pytorch model")
        
        # Else check the device
        elif device not in devices:
            raise RuntimeError(f"Processing engine not found please set device to one of the following options {devices}")
        
        # Set device
        self.device = device

        # Get model params as dictionary
        with open(os.path.join(run_dir, "model", "params.json")) as f:
            model_params = json.load(f)

        self.model_params = model_params

        # Load model on specified device and get the model params
        model_path = os.path.join(run_dir, "model", "BigramModel.pt")
        self.model = LanguageModel(vocab_size=len(encoder_hash_map.keys()), 
                                block_size=model_params['block_size'], 
                                n_embedd=model_params['n_embedd'],
                                device=device,
                                attention_head_size=model_params['attention_head_size'],
                                num_heads= model_params['num_heads'],
                                num_layers=model_params['num_layers'],
                                dropout = 0
                                )
        self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
        self.model = self.model.to(device=device)

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
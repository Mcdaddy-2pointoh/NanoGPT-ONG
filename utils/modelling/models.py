# Imports
import torch 
import torch.nn as nn
from torch.nn import functional as F
from utils.modelling.layers.transformer_block import Block
from utils.modelling.encoding.sinusoidal_positional_encoding import SinusoidalPositionalEncoding


# Model class
class LanguageModel(nn.Module):
    
    def __init__(self, vocab_size: int, block_size: int, n_embedd: int = 32, device: str = None, attention_head_size: int = 32, num_heads: int = 4, num_layers: int = 6, dropout: float = 0.2, positional_encoder_type: str = "sinusoidal"):
        """
        Function: Instances an object of class `LanguageModel`
        Args:
            vocab_size (int): Number of unique tokens mapped in the tokenizer
            block_size (int): Block size is the maximum context window of the model
            n_embedd (int): Linear dimension in which the token in projected into
            device (str): The device on which the operation must be carried out `cuda` or `cpu`
            attention_head_size (int): The projection dimension of all attention heads combined
            num_heads (int): Number of parallel heads to implement,
            positional_encoder_type (Enum(str)): Either has conventional linear postional encoding or sinusoidal positional encoding
        """
        super().__init__()

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device in ["cuda", "cpu"]:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Updating the selected device variable
        device = self.device

        # Setting positional encoders
        if not isinstance(positional_encoder_type, str):
            raise TypeError("Argument `positional_encoder_type` must be of type str")
        
        # Set `positional_encoder_type` to naive
        elif positional_encoder_type == "naive":
            self.positional_encoder_type = "naive"
            self.position_embedding_table = nn.Embedding(block_size, n_embedd).to(device=device)

        # Set `positional_encoder_type` to sinusoidal
        elif positional_encoder_type == "sinusoidal":
            self.positional_encoder_type = "sinusoidal"

        # Else key `positional_encoder_type` is out of bounds raise error
        else:
            raise ValueError("Argument `positional_encoder_type` must be either 'sinusoidal' or 'naive'")

        # Validating model params
        if not isinstance(n_embedd, int):
            try:
                n_embedd = int(n_embedd)
            except Exception as e:
                raise TypeError("Argument `n_embedd` must be of type int.")  from e
            
        if not isinstance(block_size, int):
            try:
                block_size = int(block_size)
            except Exception as e:
                raise TypeError("Argument `block_size` must be of type int.")  from e
            
        if not isinstance(vocab_size, int):
            try:
                vocab_size = int(vocab_size)
            except Exception as e:
                raise TypeError("Argument `vocab_size` must be of type int.")  from e
        
        # Setting up model params
        self.n_embedd = n_embedd
        self.block_size = block_size
        self.vocab_size = vocab_size

        # Setting the Embedding layers
        self.token_embedding_table = nn.Embedding(vocab_size, n_embedd).to(device=device)

        # Validating attention params
        if not isinstance(attention_head_size, int):
            try:
                attention_head_size = int(attention_head_size)
            except Exception as e:
                raise TypeError("Argument `attention_head_size` must be of type int.")  from e
            
        if not isinstance(num_heads, int):
            try:
                num_heads = int(num_heads)
            except Exception as e:
                raise TypeError("Argument `num_heads` must be of type int.")  from e

        if attention_head_size % num_heads != 0:
            raise ValueError("Attention heads size must be a multiple of the number of attention heads")
        
        # Setting up attention params
        self.attention_head_size = attention_head_size
        self.num_heads = num_heads

        # Setting up attention heads splitting the number of attention size over the n heads
        self.blocks = nn.Sequential(
            *[Block(num_heads = num_heads, n_embedd=n_embedd, block_size=block_size, device=device, attention_head_size=attention_head_size, dropout=dropout) for _ in range(num_layers)] 
        )

        # Layer norm
        self.ln = nn.LayerNorm(attention_head_size)

        # Pass the value to the LM head to get the token out
        self.lm_head = nn.Linear(attention_head_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Function: Feed forward function of the model 
        Args:
            idx (torch.Tensor): Input params of the model 
            targets (torch.Tensor | None): Expected annotated result of the model
        """
        # Enable device 
        
        # Type cast idx
        if type(idx) != torch.Tensor:
            try:    
                idx = torch.tensor(idx, device=self.device)
            except Exception as e:
                raise TypeError(f"Could not convert type {type(idx)} to torch.Tensor")
        
        # Get shape of idx
        B, T = idx.shape

        # Embedd Tokens
        token_embeddings = self.token_embedding_table(idx) # B, T, C = (B, T, n_embedd)
        token_embeddings = token_embeddings.to(device=self.device)

        # Embedd Position
        ## If naive linear
        if self.positional_encoder_type == "naive":
            positional_indices = torch.arange(T).to(device=self.device)
            positional_embeddings = self.position_embedding_table(positional_indices) # (T,C)

        elif self.positional_encoder_type == "sinusoidal":
            positional_embeddings = SinusoidalPositionalEncoding(T=T, n_embedd=self.n_embedd, device=self.device)
        
        else:
            raise RuntimeError("Could not encode Positions, pleaase check `positional_encoder_type` in params")

        # Combine the two embeddings to get our input tensor
        x = token_embeddings + positional_embeddings

        # Sequentially run multi-headed attention multiple times
        x = self.blocks(x)

        # Layer normalising 
        x = self.ln(x)

        # Get logits
        logits = self.lm_head(x) 

        # If we don't have targets we don't provide loss
        if targets is None:
            loss = None
        
        # Else we compute and provide loss
        else:
            if type(targets) != torch.Tensor:
                try:    
                    targets = torch.tensor(targets, device=self.device)
                except Exception as e:
                    raise TypeError(f"Could not convert type {type(targets)} to torch.Tensor")
            
            # Reshape logits for the loss & return loss and logit
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, target=targets)
        return logits, loss
    
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 100):
        """
        Function: Funtion to generate the next N tokens 
        Args:
            idx (torch.Tensor): Input params of the model 
            max_new_tokens (int): Maximum number of tokens that the model should predict
        """

        if type(idx) != torch.Tensor:
            try:    
                idx = torch.tensor(idx, device=self.device)
            except Exception as e:
                raise TypeError(f"Could not convert type {type(idx)} to torch.Tensor")

        # Generating new tokens        
        for _ in range(max_new_tokens):
            
            # Condensed idx as the embedding dim for positional embeddings in equal to block size and hence the size of idx can never be gereater than block size
            idx = idx[:, -self.block_size:]

            # Make prediction using the bigrams
            logits, loss = self(idx)

            # Focus only on the last timestep
            logits = logits[:, -1, :] # [Batch, -1, Channels]

            # Apply softmax to get probabilities
            probabs = F.softmax(logits, dim=-1) # [Batch, Channels]

            # Sample from the probability distribution
            idx_next = torch.multinomial(probabs, num_samples=1)

            # Append the next index to the current sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    

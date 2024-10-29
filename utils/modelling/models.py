# Imports
import torch 
import torch.nn as nn
from torch.nn import functional as F

# Model class
class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size: int):
        super().__init__()
        # A wrapper around the token lookup table of size vocab_size x vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Function: Feed forward function of the model 
        Args:
            idx (torch.Tensor): Input params of the model 
            targets (torch.Tensor | None): Expected annotated result of the model
        """
        # Type cast idx
        if type(idx) != torch.Tensor:
            try:    
                idx = torch.tensor(idx, device=self.device)
            except Exception as e:
                raise TypeError(f"Could not convert type {type(idx)} to torch.Tensor")
        
        # Embedd logits
        logits = self.token_embedding_table(idx)

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
    
    def generate(self, idx, max_new_tokens):
        # Type cast idx
        if type(idx) != torch.Tensor:
            try:    
                idx = torch.tensor(idx, device=self.device)
            except Exception as e:
                raise TypeError(f"Could not convert type {type(idx)} to torch.Tensor")

        # Generating new tokens        
        for _ in range(max_new_tokens):

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



    

# Imports
import torch 
import torch.nn as nn
from torch.nn import functional as F

# Model class
class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        # A wrapper around the token lookup table of size vocab_size x vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        # If we don't hae targets we don't provide loss
        if targets is None:
            loss = None
        
        # Else we compute and provide loss
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, target=targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):

            # Make prediction using the bigrams
            logits, loss = self(max_new_tokens)

            # Focus only on the last timestep
            logits = logits[:, -1, :] # [Batch, -1, Channels]

            # Apply softmax to get probabilities
            probabs = F.softmax(logits, dim=-1) # [Batch, Channels]

            # Sample from the probability distribution
            idx_next = torch.multinomial(probabs, num_samples=1)

            # Append the next index to the current sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx



    

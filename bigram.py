import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 32 # How many independent sequences we process in parallel
block_size = 8 # What is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# -------------------------

# Get data
with open('./data/ong.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
# Unique characters in the text 
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Encoder and Decoder
stoi = {ch:i for i ,ch in enumerate(chars)}
itos = {i:ch for i ,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# Create the Train and Test splits
data = torch.tensor(encode(text), dtype=torch.long)
val_split_boundary = 0.9
train_data = data[:int(val_split_boundary*len(data))]
val_data = data[int(val_split_boundary*len(data)):]

# Data Loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
        return out
    
# Basic bi gram model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # (B, T, C) tensor
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # needs a (B, C, T) tensor
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :] # Becomes (B, C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1) # Becomes (B, T+1) for max_n_tokens
        return idx
    
model = BigramLanguageModel(vocab_size=vocab_size)
m = model.to(device)

# PyTorch Optimizer 
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):

    # Evaluate Loss on train and val
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'Step {iter+1} : train loss {losses['train']:.4f} validation loss {losses['val']:.4f}')

    # Sample a batch of data
    xb, yb = get_batch('train')

    # evalute the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# Generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
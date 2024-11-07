from utils.tokenizers.naive_tokenizer import naive_tokenizer
import json
import os

# load 
path = "./runs/run-0009 (wiki-2500)"

# Load hashmaps
with open(os.path.join(path, "tokenizer", 'encoding_hashmap.json')) as f:
    encoder_hash_map = json.load(f)

with open(os.path.join(path, "tokenizer", 'decoding_hashmap.json')) as f:
    decoder_hash_map = json.load(f)

with open(os.path.join(path, "tokenizer", 'characters.json')) as f:
    characters = json.load(f)

tokenizer = naive_tokenizer
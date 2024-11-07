from utils.tokenizers.tiktokenizer import tiktokenizer

tokenizer = tiktokenizer()

text = "Hello World!"

# Encode
token_stream = tokenizer.encode(text)
print(token_stream)

# Decode
text_decoded = tokenizer.decode(tokens=token_stream)
print(text_decoded)
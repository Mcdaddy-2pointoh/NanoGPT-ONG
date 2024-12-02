from tokenizers.tiktokenizer import tiktokenizer

tokenizer = tiktokenizer()

text = "Hi, Sharvil how are you."

# Encode
token_stream = tokenizer.encode(text)
print(token_stream)

# Decode
text_decoded = tokenizer.decode(tokens=token_stream)
print(text_decoded)
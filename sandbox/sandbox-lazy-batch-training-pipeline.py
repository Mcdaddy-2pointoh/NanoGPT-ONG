from pipelines.training.lazy_batch_training import lazy_batch_training

# Data Params
data_path = "./data_archive/ROOTS.txt"
file_splitter_params = {
    'segment_target_dir' : './data_archive/segments/', 
    'array_target_dir' : './data_archive/arrays/', 
    'split_threshold' : 200_000, 
    'verbose' : False, 
    'file_encoding' : 'utf-8', 
    'write_frequency' : 2_000
}

# Tokenizer params
tokenizer_encoding = "cl100k_base"
tokenizer_vocab_size = 100_280

# Model Params  'block_size', 'n_embedd', 'attention_head_size', 'num_heads', 'num_layers', 'dropout', 'positional_encoder_type'
model_params = {
    'block_size' : 256,
    'n_embedd' : 128,
    'attention_head_size': 128,
    'num_heads': 4,
    'num_layers' : 6,
    'dropout': 0.20,
    'positional_encoder_type' : 'sinusoidal'
}

# Model check pointing params
check_point_params = {
    'save_steps' : 20
}

# Training params
training_params = {
    "learning_rate" : 1e-3,
    "batch_size" : 12,
    "steps" : 100
}

# Computational Device
device = "cuda:0"


lazy_batch_training(
    data=data_path,
    file_splitter_params=file_splitter_params,
    tokenizer_encoding=tokenizer_encoding,
    tokenizer_vocab_size=tokenizer_vocab_size,
    model_params=model_params,
    training_params=training_params,
    check_point_params=check_point_params,
    segment_data=True,
    device=device
)
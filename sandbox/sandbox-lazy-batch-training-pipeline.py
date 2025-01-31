from pipelines.training.lazy_batch_training import lazy_batch_training 
import torch

# Data Params
data_path = "./data_archive/wikipedia_2500000.txt"
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

# Model Params  'block_size', 'n_embedd', 'attention_size', 'num_heads', 'num_layers', 'dropout', 'positional_encoder_type'
model_params = {
    'block_size' : 512,
    'n_embedd' : 128,
    'attention_size': 128,
    'num_heads': 8,
    'num_layers' : 10,
    'dropout': 0.20,
    'positional_encoder_type' : 'RoPE', 
    'model_precision': "bfloat16"
}

# Model check pointing params
check_point_params = {
    'save_steps' : 1_500,
    "log_to_mlflow" : False,
    "mlflow_experiment_name": "",
    "mlflow_tracking_uri" :  "",
    "mlflow_model_name" : "",
    "dataset_name" : ""
}

# Training params
training_params = {
    "learning_rate" : 0.001,
    "batch_size" : 10,
    "steps" : 15_000,
    "lr_scheduler_type": "CosineAnnealingWarmRestarts",
    "lr_scheduler_params" : {
        "T_0": 300,
        "T_mult": 2,
        "eta_min": 1e-5
    }
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
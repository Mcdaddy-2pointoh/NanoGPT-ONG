from pipelines.training.continued_training import continued_pretrainer 
import torch

# Data Params
array_directory = "C:/Users/sharv/Documents/Sharvil/Projects/NanoGPT-ONG/data_archive/arrays"

# Model Params  'block_size', 'n_embedd', 'attention_size', 'num_heads', 'num_layers', 'dropout', 'positional_encoder_type'
model_checkpoint_path = "C:/Users/sharv/Documents/Sharvil/Projects/NanoGPT-ONG/runs/run-0035/checkpoints/LanguageModel-checkpoint-00001.pt"
optimizer_checkpoint_path = "C:/Users/sharv/Documents/Sharvil/Projects/NanoGPT-ONG/runs/run-0035/checkpoints/Optimizer-checkpoint-00001.pt"
params_path = "C:/Users/sharv/Documents/Sharvil/Projects/NanoGPT-ONG/runs/run-0035/model/params.json"

# Model check pointing params
check_point_params = {
    'save_steps' : 500,
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
    "steps" : 5_000,
    "lr_scheduler_type": "CosineAnnealingWarmRestarts",
    "lr_scheduler_params" : {
        "T_0": 300,
        "T_mult": 2,
        "eta_min": 1e-5
    }
}

# Computational Device
device = "cuda:0"


continued_pretrainer(
    model_checkpoint_path = model_checkpoint_path,
    optimizer_checkpoint_path  = optimizer_checkpoint_path,
    params_path = params_path,
    training_params = training_params,
    array_directory = array_directory,
    check_point_params=check_point_params
)
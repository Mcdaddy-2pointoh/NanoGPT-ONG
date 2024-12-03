from pipelines.training.naive_training import training_pipeline

# Training pipeline
results = training_pipeline(dir_path="./data/",
              block_size=256,
              batch_size=12,
              steps=100,
              split_ratio= 0.80,
              save_loss_curves=True,
              learning_rate=3e-4,
              max_tokens=1024,
              n_embedd = 128,
              num_heads= 4,
              attention_size = 128,
              num_layers= 6,
              dropout= 0.20,
              tokenizer_type= "tiktoken",
              tokenizer_encoding= "cl100k_base",
              runs_dir='./runs',
              smoothen_loss_plots = True,
              positional_encoder_type="sinusoidal"
              )

# Visualise
print(results["preds"])
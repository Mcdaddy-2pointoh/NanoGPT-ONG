import numpy as np
import os
import torch

# Print min loss
x = np.load("C:/Users/sharv/Documents/Sharvil/Projects/NanoGPT-ONG/runs/run-0025/metric-logs/losses.npy")
print(np.min(x))

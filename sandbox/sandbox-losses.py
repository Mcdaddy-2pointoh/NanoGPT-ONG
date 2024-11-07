import numpy as np
import os
import torch

# Print min loss
x = np.load("./runs/run-0003/loss-logs/losses.npy")
print(np.min(x))

# Print dir contents
print(os.listdir("./runs/run-0009 (wiki-2500)/model"))
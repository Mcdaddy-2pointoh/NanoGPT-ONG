import numpy as np
import os
import torch
from matplotlib import pyplot as plt

# Print min loss
x = np.load("C:/Users/sharv/Documents/Sharvil/Projects/NanoGPT-ONG/runs/run-0029/metric-logs/lr.npy")

plt.plot(list(range(1,len(x)+1)), x)
plt.savefig("./lr.png")
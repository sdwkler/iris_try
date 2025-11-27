import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_plot(xs, ys, path, title="Training"):
    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.savefig(path)
    plt.close()
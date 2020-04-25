import numpy as np
import argparse
import torch
from data import get_train_loaders, preprocess
from model import TrafficSignNet
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm
import utils
from train import fit 
from test import evaluate

# Data Initialization and Loading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
preprocess('data')
train_loader, valid_loader = get_train_loaders(
        'data', device, 64, 0, 20000)


model = torch.load('initial.ptmodel').to(device)
print("nonzeors before pruning:")
utils.print_nonzeros(model)

print("-----------Begin Prune-----------")
s = 0.65
model.prune_by_std(s)
print("nonzeros after pruning:")
utils.print_nonzeros(model)
torch.save(model, "model_after_prune.ptmodel")


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

# Data Initialization and Loading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
preprocess('data')
train_loader, valid_loader = get_train_loaders(
'data', device, 64, 0, 20000)

# Neural Network and Optimizer
model = TrafficSignNet().to(device)
criterion = nn.CrossEntropyLoss()
# reset optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# load model after pruning:
model = torch.load('model_after_prune.ptmodel')
fit(5, model, criterion, optimizer, train_loader, valid_loader, 5, 'model_after_retrain.ptmodel')



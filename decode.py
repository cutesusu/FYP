
from basic.huffmancoding import huffman_decode_model
import numpy as np
import argparse
import torch
from data import get_train_loaders, preprocess
from model import TrafficSignNet
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm
import utils
from test import evaluate
from data import get_test_loader
from torchvision.utils import make_grid
from train import valid_batch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TrafficSignNet().to(device)
utils.print_nonzeros(model)
huffman_decode_model(model, 'encodings/')
utils.print_nonzeros(model)
torch.save(model,'recover.ptmodel')


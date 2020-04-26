import utils
import os
import torch
import math
import numpy as np
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torchvision import datasets, transforms
"""
model = torch.load('initial_21.ptmodel')
utils.print_nonzeros(model)
print("---------------------------------------------")
model = torch.load('model_after_prune.ptmodel')
utils.print_nonzeros(model)
print('---------------------------------------------')
model = torch.load('model_after_retrain.ptmodel')
utils.print_nonzeros(model)
print('---------------------------------------------')
model = torch.load('model_after_weight_sharing.ptmodel')
utils.print_nonzeros(model)
print('---------------------------------------------')
"""

model = torch.load('recover.ptmodel')
utils.print_nonzeros(model)

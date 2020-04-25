import argparse

import torch

from basic.huffmancoding import huffman_encode_model


model = torch.load('model_after_weight_sharing.ptmodel')

huffman_encode_model(model)


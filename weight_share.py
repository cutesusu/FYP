import argparse
import os

import torch

from basic.quantization import apply_weight_sharing


# here model_dir is the path to saved pruned model
def WS(model_dir):
	# Define the model
        model = torch.load(model_dir)
        # Weight sharing
        print("=====Begin Weight Sharing======")
        apply_weight_sharing(model)  
        device = torch.device('cuda')
        model.to(device)
        torch.save(model, 'model_after_weight_sharing.ptmodel')


WS("model_after_retrain.ptmodel")

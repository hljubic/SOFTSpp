import argparse
import os
import random

import numpy as np
import pandas as pd
import torch

from exp.exp_custom import Exp_Custom


# fix seed for reproducibility
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
torch.set_num_threads(6)

# basic config
config = {
    # dataset settings
    'root_path': './dataset/ETT-small/',
    'data_path': 'ETTm1.csv',
    'data': 'ETTm1',
    'features': 'M',
    'freq': 'h',
    'seq_len': 96,
    'pred_len': 96,
    # model settings
    'model': 'HASPFormer',
    'checkpoints': './checkpoints/',
    'd_model': 128,
    'd_core': 64,
    'd_ff': 128,
    'e_layers': 2,
    'learning_rate': 0.0003,
    'lradj': 'cosine',
    'train_epochs': 50,
    'patience': 3,
    'batch_size': 16,
    'dropout': 0.0,
    'activation': 'gelu',
    'use_norm': True,
    # system settings
    'num_workers': 0,
    'use_gpu': True,
    'gpu': '0',
    'save_model': True,
}

parser = argparse.ArgumentParser(description='HASPFormer')
args = parser.parse_args([])
args.__dict__.update(config)
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

print('Args in experiment:')
print(args)
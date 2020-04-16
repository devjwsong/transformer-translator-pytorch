from tqdm import tqdm
from src.constants import *
from src.custom_data import *
from src.transformer import *

import torch


class Manager():
    def __init__(self):
        # Load dataloaders
        self.train_loader, self.valid_loader, self.test_loader = get_data_loader()

        # Load each vocab dict
        

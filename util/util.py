import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
from sklearn import metrics
import time
import random
import pandas as pd
from einops import rearrange, repeat

from sklearn.preprocessing import label_binarize

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)  # gpu

class Early_Stop():
    def __init__(self, patience=7):
        self.best_loss = np.inf
        self.model_state_dict = None
        self.count = 0
        self.patience = patience
        
    def __call__(self, loss, model):

        if loss < self.best_loss:
            self.best_loss = loss
            self.count = 0
            self.model_state_dict = model.state_dict()
        else:
            self.count += 1

        if self.count > self.patience:
            print("early stoping...")
            return True
        
        return False
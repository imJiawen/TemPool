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
    
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, save_path=None, dp_flag=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path
        self.dp_flag = dp_flag

    def __call__(self, val_loss, model, classifier=None, time_predictor=None, decoder=None):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience} ({self.val_loss_min:.6f} --> {val_loss:.6f})')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        model_state_dict = model.state_dict()

        if self.save_path is not None:
            torch.save({
                'model_state_dict':model_state_dict
            }, self.save_path)
        else:
            print("no path assigned")  

        self.val_loss_min = val_loss
        
def evaluate_mc(label, pred, n_class=6):
    if n_class > 2:
        labels_classes = label_binarize(label, classes=range(n_class))
        pred_scores = pred
        idx = np.argmax(pred_scores, axis=-1)
        preds_label = np.zeros(pred_scores.shape)
        preds_label[np.arange(preds_label.shape[0]), idx] = 1
        acc = metrics.accuracy_score(labels_classes, preds_label)
    else:
        labels_classes = label
        pred_scores = pred[:, 1]
        acc = np.mean(pred.argmax(1) == label)

    auroc = metrics.roc_auc_score(labels_classes, pred_scores, average='macro')
    auprc = metrics.average_precision_score(labels_classes, pred_scores, average='macro')

    return acc, auroc, auprc

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
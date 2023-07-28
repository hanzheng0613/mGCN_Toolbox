# Test file for toolbox

import argparse
from OpenAttMultiGL.model.mGCN.mGCN_link import*
import torch.nn as nn
import torch.optim as optim
import torch
from OpenAttMultiGL.utils.dataset import dataset
from OpenAttMultiGL.utils.process import * #split_link_data
from sklearn.metrics import roc_auc_score
import numpy as np
import random
import copy
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score

parser = argparse.ArgumentParser(description='GCN')

parser.add_argument('--dataset', type=str, default='amazon')
parser.add_argument('--fast_split', action='store_true',
                    help="for large custom datasets (not OGB), do a fast data split")

parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--epochs', type=int, default=20,
                    help='Number of training epochs.')
parser.add_argument('--alpha', type=float, default=0.6,
                    help='Hyperparameter')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout')
parser.add_argument('--training_ratio', type=float, default=0.3,
                    help='Training Ratio')
parser.add_argument('--validing_ratio', type=float, default=0.1,
                    help='Validing Ratio')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning Rate')
parser.add_argument('--weight_decay', type=float, default=1e-2,
                    help='Weight_decay')
parser.add_argument('--test_view', type=int, default=1,
                    help='Number of training epochs.')
parser.add_argument('--neg_k', type=int, default=1,
                    help='Number of negative samples.')
parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")

args = parser.parse_args()



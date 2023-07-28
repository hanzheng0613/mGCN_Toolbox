# Test file for toolbox

import argparse
from OpenAttMultiGL.model.mGCN.mGCN_node import*
import torch.nn as nn
import torch.optim as optim
import torch
from OpenAttMultiGL.utils.dataset import dataset
from OpenAttMultiGL.utils.process import split_node_data
from sklearn.metrics import roc_auc_score
import numpy as np
import random
import copy
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score

parser = argparse.ArgumentParser(description='GCN')

parser.add_argument('--dataset', type=str, default='acm')
parser.add_argument('--fast_split', action='store_true',
                    help="for large custom datasets (not OGB), do a fast data split")

parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of training epochs.')
parser.add_argument('--alpha', type=float, default=0.6,
                    help='Hyperparameter')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout')
parser.add_argument('--training_ratio', type=float, default=0.3,
                    help='Training Ratio')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning Rate')
parser.add_argument('--weight_decay', type=float, default=1e-2,
                    help='Weight_decay')
parser.add_argument('--test_view', type=int, default=1,
                    help='Number of training epochs.')
parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")

args = parser.parse_args()


def get_accuracy(output, labels):
    """
    Accuracy calculation method
    """
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def train():
    model_mGCN.train()
    optimizer.zero_grad()
    out = model_mGCN(sample_data.dataset)
    loss_train = F.cross_entropy(out[training_id], sample_data.labels[training_id])
    acc_train = get_accuracy(out[training_id], sample_data.labels[training_id])
    
    loss_train.backward()
    optimizer.step()
    

    
    loss_val = F.cross_entropy(out[valid_id], sample_data.labels[valid_id])
    acc_val = get_accuracy(out[valid_id], sample_data.labels[valid_id])
    
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()))

@torch.no_grad()
def test():
    model_mGCN.eval()
    out = model_mGCN(sample_data.data)

    loss_test = F.nll_loss(out[test_id], sample_data.labels[test_id])
    acc_test = get_accuracy(out[test_id], sample_data.labels[test_id])
    
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    
sample_data = dataset(args.dataset)

taskname  = 'node'

training_id, valid_id, test_id = split_node_data(len(sample_data.labels),train_percent=args.training_ratio,valid_percent = 0.1)
#data, num_views, training_id, valid_id, test_id, num_classes, labels, adj_list, edge_list = load_data(args.dataset, training_percent=args.training_ratio)
# data, split_edges = split_data(data_ori, args.test_view, multi=True)
print("Finish loading data")
num_feat = sample_data.dataset.x.shape[1]
model_mGCN = mGCN(num_feat, args.hidden, None, sample_data.num_dims, args.alpha, sample_data.num_classes, dropout=args.dropout)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model_mGCN.parameters(), lr=args.lr)

    
for epoch in range(args.epochs):
    train()
    
print("Model training is complete")

test()
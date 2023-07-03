# Test file for toolbox

import argparse
from mGCN_Toolbox.model.mGCN.mGCN_link import*
import torch.nn as nn
import torch.optim as optim
import torch
from mGCN_Toolbox.utils.dataset import dataset
from mGCN_Toolbox.utils.process import * #split_link_data
from sklearn.metrics import roc_auc_score
import numpy as np
import random
import copy
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score

parser = argparse.ArgumentParser(description='GCN')

parser.add_argument('--dataset', type=str, default='dblp')
parser.add_argument('--fast_split', action='store_true',
                    help="for large custom datasets (not OGB), do a fast data split")

parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--epochs', type=int, default=100,
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


def area_under_prc(pred, target):
    """
    Area under precision-recall curve (PRC).
    Parameters:
        pred (Tensor): predictions of shape :math:`(n,)`
        target (Tensor): binary targets of shape :math:`(n,)`
    """
    order = np.argsort(-1 * pred) #pred.argsort(descending=True) 
    target = target[order]
    print(type(target))
    precision = target.cumsum(0) / torch.arange(1, len(target) + 1, device=target.device)
    auprc = precision[target == 1].sum() / ((target == 1).sum() + 1e-10)
    return auprc

def evaluate_metrics(true, pred, test, positive_num):

    AUC_value = roc_auc_score(true, pred)
    if test:
        pred = torch.FloatTensor(pred)
        true = torch.FloatTensor(true)
        ap = area_under_prc(pred, true)
        
        # hits = eval_hits(pred[0:positive_num], pred[positive_num:], 'numpy')
        # pred = torch.FloatTensor(pred)
        # hits = eval_precision(pred, positive_num, 'torch')
        return AUC_value,ap
    return AUC_value

def evaluate_model(edge, edge_neg, common_neighbors, labels, test=True, num_pos=None):
    model_GNN.eval()
    logits = model_GNN(data, edge, edge_neg, args.test_view)

    return evaluate_metrics(labels.cpu().numpy(), torch.sigmoid(logits).cpu().detach().numpy(), test, edge.shape[0])

best_val = 0
best_test = 0
best_hits = [0, 0, 0, 0, 0, 0, 0]
def replace(best_hits, temp):
    results = []
    for i in range(0, len(best_hits)):
        if temp[i] > best_hits[i]:
            results.append(temp[i])
        else:
            results.append(best_hits[i])
    return results

def train_model(epochs):
    global best_val
    global best_test
    global best_hits
    best_val = 0
    best_test = 0
    best_hits = [0, 0, 0, 0, 0, 0, 0]
    # print(split_edges['train']['edge'].shape[0])
    # training_negative = split_edges['train']['edge_neg'][range(0, split_edges['train']['edge'].shape[0])]
    training_negative = []
    labels = []
    training_positive = []
    for i in range(0, sample_data.num_dims):
        training_negative.append(split_edges[i]['train']['edge_neg'][np.random.randint(0, split_edges[i]['train']['edge_neg'].shape[0], split_edges[i]['train']['edge'].shape[0])])
        labels.append(split_edges[i]['train']['label'])
        training_positive.append(split_edges[i]['train']['edge'])
    for epoch in range(0, epochs):
        model_GNN.train()
        optimizer.zero_grad()
        # print(training_negative.shape)
        logits = model_GNN(data, training_positive, training_negative)
        # labels = split_edges[i]['train']['label']
        loss_list = [criterion(logit, label) for logit, label in zip(logits, labels)]
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        #print(type(split_edges[args.test_view]['valid']['edge_neg'].shape))
        #print(split_edges[args.test_view]['valid']['edge_neg'].shape)
        eval,ap = evaluate_model(split_edges[args.test_view]['valid']['edge'], split_edges[args.test_view]['valid']['edge_neg'], None,
                              split_edges[args.test_view]['valid']['label'])
        # if epoch == epochs-1:
        #     temp = evaluate_model(split_edges['test']['edge'], split_edges['test']['edge_neg'], None,
        #                    split_edges['test']['label'], test=True)
        #     print("AUC last:", temp)
        auc_list = []
        ap_list = []
        print("AUC: ",eval)
        print("Precision: ", ap)
        auc_list.append(eval)
        ap_list.append(ap)
        if eval > best_val:
            best_val = eval
            best_test,best_ap = evaluate_model(split_edges[args.test_view]['test']['edge'], split_edges[args.test_view]['test']['edge_neg'], None,
                                       split_edges[args.test_view]['test']['label'], test=True)
            # best_hits = replace(best_hits, temp)
            # print(best_hits)
            print('Epoch:', epoch)
            print("Best Validation:", best_val)
            print("Best Test:", best_test)
            print("Best Ap:", best_ap)


results = []
results_hits = []
# results_hits = {}
# for K in [20, 50, 100]:
#     results_hits[f'Hits@{K}'] = []

for run in range(0, args.runs):
    np.random.seed(run)
    torch.manual_seed(run)
    torch.cuda.manual_seed(run)
    random.seed(run)

    sample_data= dataset(args.dataset)
    data, split_edges = split_link_data(sample_data.dataset, args.test_view, args.neg_k, multi=True)
    print("Finish loading data")
    num_feat = data.x.shape[1]
    model_GNN = mGCN(num_feat, args.hidden, None, sample_data.num_dims, args.alpha, dropout=args.dropout)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model_GNN.parameters(), lr=args.lr)

    data.x = data.x#.cuda()
    for i in range(0, len(data.edge_index)):
        data.edge_index[i] = data.edge_index[i]#.cuda()
    for i in range(0, len(split_edges)):

        split_edges[i]['train']['label'] = torch.cat(
            (split_edges[i]['train']['label_pos'], split_edges[i]['train']['label_neg']))#.cuda()
        if i == args.test_view:
            split_edges[i]['valid']['label'] = torch.cat(
                (split_edges[i]['valid']['label_pos'], split_edges[i]['valid']['label_neg']))#.cuda()
            split_edges[i]['test']['label'] = torch.cat(
                (split_edges[i]['test']['label_pos'], split_edges[i]['test']['label_neg']))#.cuda()
    model_GNN#.cuda()
    best_val = 0
    best_test = 0
    # results_hits.append(best_hits)
    train_model(args.epochs)
    results.append(best_test)
    # for K in [20, 50, 100]:
    #     results_hits[f'Hits@{K}'].append(best_hits[f'Hits@{K}'])

print(f'   Final Test: {np.mean(results):.4f} ± {np.std(results):.4f}')
import argparse
from utils import load_data
from model import mGCN
import torch.nn as nn
import torch.optim as optim
import torch
from sklearn.metrics import roc_auc_score
import numpy as np
import random
import copy
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score

parser = argparse.ArgumentParser(description='GCN')

parser.add_argument('--dataset', type=str, default='acm')
parser.add_argument('--fast_split', action='store_true',
                    help="for large custom datasets (not OGB), do a fast data split")

parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--epochs', type=int, default=500,
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

args = parser.parse_args()

def evaluate_metrics(true, pred):
    preds = pred.max(1)[1].type_as(labels)
    correct = preds.eq(true).double()
    correct = correct.sum()
    return correct /len(true)

def evaluate_model(ind):
    model_GNN.eval()
    logits = model_GNN(data)
    f1_test = f1_score(labels[ind].cpu().numpy(), logits.max(1)[1][ind].detach().cpu().numpy(),
                       average="macro")
    return f1_test
    # return evaluate_metrics(labels[ind], logits[ind]).item()
soft = nn.Softmax(dim=1)
def write_results(ind):
    model_GNN.eval()
    logits = soft(model_GNN(data))
    f= open("./results/" + args.dataset + "_combined", "w")
    for temp in logits[ind].detach().cpu().numpy():
        f.write(" ".join(np.array([str(i) for i in temp])) +"\n")
    f.close()

best_val = 0
best_test = 0

def train_model(epochs):
    global best_val
    global best_test
    global best_hits
    best_val = 0
    best_test = 0
    # print(split_edges['train']['edge'].shape[0])
    # training_negative = split_edges['train']['edge_neg'][range(0, split_edges['train']['edge'].shape[0])]
    training_negative = []
    training_positive = []

    for epoch in range(0, epochs):
        model_GNN.train()
        optimizer.zero_grad()
        # print(training_negative.shape)
        logits = model_GNN(data)
        # labels = split_edges[i]['train']['label']
        loss = criterion(logits[training_id], labels[training_id])
        loss.backward()
        optimizer.step()

        eval = evaluate_model(valid_id)
        # if epoch == epochs-1:
        #     temp = evaluate_model(split_edges['test']['edge'], split_edges['test']['edge_neg'], None,
        #                    split_edges['test']['label'], test=True)
        #     print("AUC last:", temp)

        if eval > best_val:
            best_val = eval
            best_test = evaluate_model(test_id)
            # write_results(test_id)
            print('Epoch:', epoch)
            print("Best Validation:", best_val)
            print("Best Test:", best_test)


results = []
# results_hits = {}
# for K in [20, 50, 100]:
#     results_hits[f'Hits@{K}'] = []
for run in range(0, args.runs):
    np.random.seed(run)
    torch.manual_seed(run)
    torch.cuda.manual_seed(run)
    random.seed(run)

    data, num_views, training_id, valid_id, test_id, num_classes, labels, adj_list = load_data(args.dataset, training_percent=args.training_ratio)
    # data, split_edges = split_data(data_ori, args.test_view, multi=True)
    print("Finish loading data")
    num_feat = data.x.shape[1]
    model_GNN = mGCN(num_feat, args.hidden, None, num_views, args.alpha, num_classes, dropout=args.dropout)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model_GNN.parameters(), lr=args.lr)

    data.x = data.x.cuda()
    for i in range(0, len(data.edge_index)):
        data.edge_index[i] = data.edge_index[i].cuda()
    training_id = training_id.cuda()
    valid_id = valid_id.cuda()
    test_id = test_id.cuda()
    labels = labels.cuda()

    # for i in range(0, len(split_edges)):
    #
    #     split_edges[i]['train']['label'] = torch.cat(
    #         (split_edges[i]['train']['label_pos'], split_edges[i]['train']['label_neg'])).cuda()
    #     if i == args.test_view:
    #         split_edges[i]['valid']['label'] = torch.cat(
    #             (split_edges[i]['valid']['label_pos'], split_edges[i]['valid']['label_neg'])).cuda()
    #         split_edges[i]['test']['label'] = torch.cat(
    #             (split_edges[i]['test']['label_pos'], split_edges[i]['test']['label_neg'])).cuda()
    model_GNN.cuda()
    best_val = 0
    best_test = 0
    best_hits = {}
    train_model(args.epochs)
    results.append(best_test)
    # for K in [20, 50, 100]:
    #     results_hits[f'Hits@{K}'].append(best_hits[f'Hits@{K}'])

print(f'   Final Test: {np.mean(results):.4f} ± {np.std(results):.4f}')
# for K in [20, 50, 100]:
#     print(f'   "Hits@"+str(K): {np.mean(results_hits["Hits@"+str(K)]):.4f} ± {np.std(results_hits["Hits@"+str(K)]):.4f}')
# f = open("results/mGCN", 'a+')
# f.write(args.dataset + '_' + str(args.training_ratio) + '_mGCN' +f'   Final Test: {np.mean(results):.4f} ± {np.std(results):.4f}\n')
# f.close()
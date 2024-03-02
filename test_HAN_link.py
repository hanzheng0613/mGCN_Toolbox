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
from OpenAttMultiGL.model.HAN.HAN import *
parser = argparse.ArgumentParser(description='GCN')

parser.add_argument('--dataset', type=str, default='acm')
parser.add_argument('--fast_split', action='store_true',
                    help="for large custom datasets (not OGB), do a fast data split")

parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--hidden_units', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--epochs', type=int, default=1,
                    help='Number of training epochs.')
parser.add_argument('--alpha', type=float, default=0.6,
                    help='Hyperparameter')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout')
parser.add_argument('--training_ratio', type=float, default=0.3,
                    help='Training Ratio')
parser.add_argument('--validing_ratio', type=float, default=0.1,
                    help='Validing Ratio')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Learning Rate')
parser.add_argument('--weight_decay', type=float, default=0.001,
                    help='Weight_decay')
parser.add_argument('--test_view', type=int, default=0,
                    help='Number of training epochs.')
parser.add_argument('--neg_k', type=int, default=1,
                    help='Number of negative samples.')
parser.add_argument('--num_heads', type=list, default=[8],
                    help='Number of head.')

parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                    help='Device.')
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
    #print(type(target))
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
    model_HAN.eval()
    logits = model_HAN(data, edge, edge_neg, args.test_view)
    #logits = torch.stack(logits)
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
    best_ap = 0
    best_auc = 0
    auc_list = []
    ap_list = []
    # print(split_edges['train']['edge'].shape[0])
    # training_negative = split_edges['train']['edge_neg'][range(0, split_edges['train']['edge'].shape[0])]
    training_negative = []
    labels = []
    training_positive = []
    for i in range(0, c.num_dims):
        training_negative.append(split_edges[i]['train']['edge_neg'][np.random.randint(0, split_edges[i]['train']['edge_neg'].shape[0], split_edges[i]['train']['edge'].shape[0])])
        labels.append(split_edges[i]['train']['label'])
        training_positive.append(split_edges[i]['train']['edge'])
    for epoch in range(0, epochs):
        model_HAN.train()
        optimizer.zero_grad()
        # print(training_negative.shape)
        
        logits = model_HAN(data, training_positive, training_negative)
        
        # labels = split_edges[i]['train']['label']
        loss_list = [criterion(logit, label) for logit, label in zip(logits, labels)]
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        #print(type(split_edges[args.test_view]['valid']['edge_neg'].shape))
        #print(split_edges[args.test_view]['valid']['edge_neg'].shape)
        #auc,ap = evaluate_model(data, training_negative, None,
                              #split_edges[args.test_view]['valid']['label'])
        auc,ap = evaluate_model(split_edges[args.test_view]['valid']['edge'], split_edges[args.test_view]['valid']['edge_neg'], None,
                              split_edges[args.test_view]['valid']['label'])
        # if epoch == epochs-1:
        #     temp = evaluate_model(split_edges['test']['edge'], split_edges['test']['edge_neg'], None,
        #                    split_edges['test']['label'], test=True)
        #     print("AUC last:", temp)
        
        ap = float(ap)
        auc_list.append(auc)
        #print('ap: ', ap)
        ap_list.append(ap)
        if auc > best_auc and ap >best_ap:
            best_auc = auc
            best_ap = ap
            print('Epoch:', epoch)
            print("Best AUC:", best_auc)
            print("Best Ap:", best_ap)
    return auc_list,ap_list

results = []
results_hits = []
# results_hits = {}
# for K in [20, 50, 100]:
#     results_hits[f'Hits@{K}'] = []
dataname = args.dataset
c = dataset(dataname)
#args["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"

#print('feature',c.edge_list[0].shape)
    
    
if dataname == "amazon":
    data = pkl.load(open('OpenAttMultiGL/data/HAN/AMAZON/amazon.pkl', "rb"))
    data["IVI"] = sp.csr_matrix(data["IVI"])
    data["IBI"] = sp.csr_matrix(data["IBI"])
    data["IOI"] = sp.csr_matrix(data["IOI"])
    author_g = dgl.from_scipy(data["IVI"])
    subject_g = dgl.from_scipy(data["IBI"])
    o_g = dgl.from_scipy(data["IOI"])
    gs = [author_g, subject_g, o_g]
elif dataname == "acm":
    data = sio.loadmat('OpenAttMultiGL/data/HAN/ACM/acm.mat')
    data["PAP"] = sp.csr_matrix(data["PAP"])
    data["PLP"] = sp.csr_matrix(data["PLP"])
    author_g = dgl.from_scipy(data["PAP"])
    subject_g = dgl.from_scipy(data["PLP"])
    gs = [author_g, subject_g]
elif dataname == "dblp":
    data = pkl.load(open('OpenAttMultiGL/data/HAN/DBLP/dblp.pkl', "rb"))
    data["PAP"] = sp.csr_matrix(data["PAP"])
    data["PPrefP"] = sp.csr_matrix(data["PPrefP"])
    data["PATAP"] = sp.csr_matrix(data["PATAP"])
    author_g = dgl.from_scipy(data["PAP"])
    subject_g = dgl.from_scipy(data["PPrefP"])
    o_g = dgl.from_scipy(data["PATAP"])
    gs = [author_g, subject_g, o_g]
elif dataname == "imdb":
    data = pkl.load(open('OpenAttMultiGL/data/HAN/IMDB/imdb.pkl', "rb"))
    data["MDM"] = sp.csr_matrix(data["MDM"])
    data["MAM"] = sp.csr_matrix(data["MAM"])
    author_g = dgl.from_scipy(data["MDM"])
    subject_g = dgl.from_scipy(data["MAM"])
    gs = [author_g, subject_g]
    
num_classes = c.gcn_labels.shape[1]
c.gcn_labels = torch.from_numpy(data["label"]).long()
c.gcn_labels = c.gcn_labels.nonzero()[:, 1]
c.features = c.features.toarray()
c.features = torch.from_numpy(data["feature"]).float()
num_nodes = author_g.num_nodes()
train_mask = get_binary_mask(num_nodes, c.train_id)
val_mask = get_binary_mask(num_nodes, c.valid_id)
test_mask = get_binary_mask(num_nodes, c.test_id)
    
#t = dataset(args["dataset"])
#print(type(t.edge_index))
if hasattr(torch, "BoolTensor"):
    train_mask = train_mask.bool()
    val_mask = val_mask.bool()
    test_mask = test_mask.bool()

c.features = c.features.to(args.device)
c.gcn_labels = c.gcn_labels.to(args.device)
    
    
train_mask = train_mask.to(args.device)
val_mask = val_mask.to(args.device)
test_mask = test_mask.to(args.device)



for run in range(0, args.runs):
    np.random.seed(run)
    torch.manual_seed(run)
    torch.cuda.manual_seed(run)
    random.seed(run)


    #sample_data.dataset.x = [3025,1870]
    #sample_data.dataset.x = sample_data.dataset.x[:3025,:1870]
    #print('dataset  ',c.dataset.edge_index)
    data, split_edges = split_link_data(c.dataset, args.test_view, args.neg_k, multi=True)
    print("Finish loading data")
    num_feat = data.x.shape[1]
    #print("hidden_units",args.hidden_units)
    #print("out size", num_classes)
    #print('in size', c.features.shape[1])
    model_HAN =model = HAN(
        num_meta_paths=len(gs),
        in_size=c.features.shape[1],
        hidden_size=args.hidden_units,
        out_size=num_classes,
        num_heads=args.num_heads,
        dropout=args.dropout,
        alpha=args.alpha
        ).to(args.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model_HAN.parameters(), lr=args.lr)

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
    #model_GNN.cuda()
    best_val = 0
    best_test = 0
    # results_hits.append(best_hits)
    auc_list,ap_list = train_model(args.epochs)
    results.append(best_test)
    # for K in [20, 50, 100]:
    #     results_hits[f'Hits@{K}'].append(best_hits[f'Hits@{K}'])

print('Final Test:')
print('AUC: ', np.mean(auc_list), np.std(auc_list))
print('AP: ', np.mean(ap_list), np.std(ap_list))
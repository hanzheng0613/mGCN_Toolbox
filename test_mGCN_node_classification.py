# Test file for toolbox

import argparse
from OpenAttMultiGL.model.mGCN.mGCN_node import*
from OpenAttMultiGL.utils.process import *
from OpenAttMultiGL.layers.hdmi.gcn import GCN
from OpenAttMultiGL.model.mGCN.evaluate import evaluate
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
from sklearn.metrics import normalized_mutual_info_score, pairwise, f1_score
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser(description='GCN')
parser.add_argument('--dataset', type=str, default='amazon')
parser.add_argument('--fast_split', action='store_true',
                    help="for large custom datasets (not OGB), do a fast data split")

parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--epochs', type=int, default=1,
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

def evaluate_metrics(true, pred):
    preds = pred.max(1)[1].type_as(labels)
    correct = preds.eq(true).double()
    correct = correct.sum()
    return correct /len(true)

def evaluate_model(ind):
    model_mGCN.eval()
    
    logits = model_mGCN(sample_data.dataset)
    nb_classes = sample_data.num_classes
    
    pred = logits.max(1).indices
    macro_f1 = f1_score(sample_data.labels[ind].cpu().numpy(), logits.max(1)[1][ind].detach().cpu().numpy(),
                       average="macro")
    micro_f1 = f1_score(sample_data.labels[ind].cpu().numpy(), logits.max(1)[1][ind].detach().cpu().numpy(),
                       average="micro")
    #t = torch.LongTensor(sample_data.gcn_labels[sample_data.test_id]) 
    #test_lbls = torch.argmax(t, dim=1)
    #nmi = run_kmeans(sample_data.labels,pred, nb_classes)
    
    return macro_f1,micro_f1
    # return evaluate_metrics(labels[ind], logits[ind]).item()
soft = nn.Softmax(dim=1)

def run_kmeans(x, y, k):
    estimator = KMeans(n_clusters=k,n_init=10)

    NMI_list = []
    for i in range(10):
        estimator.fit(x)
        y_pred = estimator.predict(x)
        s = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
        NMI_list.append(s)

    mean = np.mean(NMI_list)
    std = np.std(NMI_list)
    #print('\t[Clustering] NMI: {:.4f} | {:.4f}'.format(mean, std))
    return mean

def write_results(ind):
    model_mGCN.eval()
    logits = soft(model_mGCN(sample_data.dataset))
    f= open("./results/" + args.dataset + "_combined", "w")
    for temp in logits[ind].detach().cpu().numpy():
        f.write(" ".join(np.array([str(i) for i in temp])) +"\n")
    f.close()
    
def combine_att(h_list):
    att_act1 = nn.Tanh()
    att_act2 = nn.Softmax(dim=-1)
    h_combine_list = []
    for i, h in enumerate(h_list):
        h = w_list[i](h)
        h = y_list[i](h)
        h_combine_list.append(h)
    score = torch.cat(h_combine_list, -1)
    score = att_act1(score)
    score = att_act2(score)
    score = torch.unsqueeze(score, -1)
    h = torch.stack(h_list, dim=1)
    h = score * h
    h = torch.sum(h, dim=1)
    return h

def embed(seq, adj_list, sparse):
    global w_list
    global y_list
    gcn_list = nn.ModuleList([GCN(ft_size, hid_units) for _ in range(n_networks)])
    w_list = nn.ModuleList([nn.Linear(hid_units, hid_units, bias=False) for _ in range(n_networks)])
    y_list = nn.ModuleList([nn.Linear(hid_units, 1) for _ in range(n_networks)])
    h_1_list = []
    for i, adj in enumerate(adj_list):
        h_1 = torch.squeeze(gcn_list[i](seq, adj, sparse))
        h_1_list.append(h_1)
    h = combine_att(h_1_list)
    return h.detach()

    
def run_similarity_search(true_label,pred_label):

    c = 0
        
    for i in range(len(true_label)):
        if pred_label[i] == true_label[i]:
            c += 1
        
    sim = c/len(true_label)
    return sim

best_val = 0
best_test = 0

def train_model(epochs):
    global best_val
    global best_test
    global best_hits
    best_val = 0
    best_test = 0
    best_macro = 0
    best_micro = 0
    # print(split_edges['train']['edge'].shape[0])
    # training_negative = split_edges['train']['edge_neg'][range(0, split_edges['train']['edge'].shape[0])]
    training_negative = []
    training_positive = []
    sparse = True
    macro_list = []
    micro_list = []
    k1_list = []
    sim_list = []
    nmi_list = []
    labels = torch.FloatTensor(sample_data.gcn_labels)
    idx_train = torch.LongTensor(sample_data.train_id)
    idx_val = torch.LongTensor(sample_data.valid_id)
    idx_test = torch.LongTensor(sample_data.test_id)
    for epoch in range(0, epochs):
        model_mGCN.train()
        optimizer.zero_grad()
        # print(training_negative.shape)
        logits = model_mGCN(sample_data.dataset)
        # labels = split_edges[i]['train']['label']
        loss = criterion(logits[sample_data.train_id], sample_data.labels[sample_data.train_id])
        loss.backward()
        optimizer.step()

        
        pred = logits.max(1).indices
        
        macro,micro = evaluate_model(sample_data.test_id)
        micro_list.append(micro)
        macro_list.append(macro)
        
        #sim = run_similarity_search(test_embs, test_lbls)
        #sim = sim[0]
        #sim_list.append(sim)
        #k1_list.append(k1)
        # write_results(test_id)
        if macro > best_macro and micro> best_micro:
            best_macro = macro
            best_micro = micro
            print('Epoch:', epoch)
            print("Macro_F1:", macro)
            print("Micro_F1:", micro)
            #print("NMI:", nmi)
            
    features = torch.FloatTensor(preprocessed_features)
    gcn_adj_list = [normalize_adj(adj) for adj in sample_data.gcn_adj_list]
    adj_list = [sparse_mx_to_torch_sparse_tensor(adj) for adj in gcn_adj_list]
    embeds = embed(features, adj_list, sparse)        
    print('Final result: ')
    print("Macro_F1:", best_macro)
    print("Micro_F1:", best_micro)        
    evaluate(embeds, idx_train, idx_val, idx_test, labels)
    



results = []
# results_hits = {}
# for K in [20, 50, 100]:
#     results_hits[f'Hits@{K}'] = []
for run in range(0, args.runs):
    np.random.seed(run)
    torch.manual_seed(run)
    torch.cuda.manual_seed(run)
    random.seed(run)


sample_data = dataset(args.dataset)
preprocessed_features = preprocess_features(sample_data.features)
ft_size = preprocessed_features[0].shape[1] 
hid_units = 128
n_networks = len(sample_data.adj_list)
taskname  = 'node'

#training_id, valid_id, test_id = split_node_data(len(sample_data.labels),train_percent=args.training_ratio,valid_percent = 0.1)
#data, num_views, training_id, valid_id, test_id, num_classes, labels, adj_list, edge_list = load_data(args.dataset, training_percent=args.training_ratio)
# data, split_edges = split_data(data_ori, args.test_view, multi=True)
print("Finish loading data")
num_feat = sample_data.dataset.x.shape[1]
model_mGCN = mGCN(num_feat, args.hidden, None, sample_data.num_dims, args.alpha, sample_data.num_classes, dropout=args.dropout)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model_mGCN.parameters(), lr=args.lr)
train_model(args.epochs)
evaluate_model(sample_data.valid_id)  

#for epoch in range(args.epochs):
    #train()
    
print("Model training is complete")

#test()
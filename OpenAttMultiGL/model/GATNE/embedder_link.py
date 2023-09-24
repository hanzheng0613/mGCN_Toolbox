import torch
import torch.nn as nn

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, pairwise, f1_score
from sklearn.metrics import roc_auc_score
from OpenAttMultiGL.model.GATNE.logreg_link import LogReg
import random

#import tensorflow as tf

def area_under_prc(pred, target):
    """
    Area under precision-recall curve (PRC).
    Parameters:
        pred (Tensor): predictions of shape :math:`(n,)`
        target (Tensor): binary targets of shape :math:`(n,)`
    """
    order = pred.argsort(descending=True)
    target = target[order]
    #print(type(target))
    precision = target.cumsum(0) / torch.arange(1, len(target) + 1, device=target.device)
    auprc = precision[target == 1].sum() / ((target == 1).sum() + 1e-10)
    return auprc

def eval_hits(y_pred_pos, y_pred_neg, type_info):
    '''
        compute Hits@K
        For each positive target node, the negative target nodes are the same.

        y_pred_neg is an array.
        rank y_pred_pos[i] against y_pred_neg for each i
    '''
    hits_arr = []
    K_arr = [20, 50, 100, 200, 500, 1000, 5000]
    for K in K_arr:
        if len(y_pred_neg) < K:
            hits_arr.append(1)
            continue

        if type_info == 'torch':
            kth_score_in_negative_edges = torch.topk(y_pred_neg, K)[0][-1]
            hitsK = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)

        # type_info is numpy
        else:
            kth_score_in_negative_edges = np.sort(y_pred_neg)[-K]
            hitsK = float(np.sum(y_pred_pos > kth_score_in_negative_edges)) / len(y_pred_pos)
        hits_arr.append(hitsK)

    return hits_arr

def evaluate_metrics(true, pred, test, positive_num):

    AUC_value = roc_auc_score(true, pred)
    if test:
        hits = eval_hits(pred[0:positive_num], pred[positive_num:], 'numpy')
        pred = torch.FloatTensor(pred)
        true = torch.FloatTensor(true)
        ap = area_under_prc(pred, true)
        # hits = eval_hits(pred, positive_num, 'torch')
        return AUC_value, hits, ap
    return AUC_value

def evaluate_model(model, embeds, edge, edge_neg, common_neighbors, labels, test=False, num_pos=None):
    model.eval()
    logits = model(embeds, edge, edge_neg)
    return evaluate_metrics(labels.cpu().numpy(), torch.sigmoid(logits).cpu().detach().numpy(), test, edge.shape[0])



def link_evaluate(embeds, split_edges, num_classes,isTest=True):
    training_negative = split_edges['train']['edge_neg'][np.random.randint(0, split_edges['train']['edge_neg'].shape[0], split_edges['train']['edge'].shape[0])]
    # train_embs = embeds[idx_train]
    xent = nn.BCEWithLogitsLoss()
    
    
  
    embeds = torch.from_numpy(embeds)
    #embeds = embeds.T
    #print("embeds:",embeds.shape)
    
    auc_list = []
    ap_list = []
    hits_list = []
    best_auc = 0
    best_ap = 0
    best_hits = [0, 0, 0, 0, 0, 0, 0]
    for epoch in range(1000):
        #print("initial",embeds.shape)
        #num_classes = 3
        #embeds.append(num_classes)
        embed_dim = embeds.shape[1]
        
        log = LogReg(embed_dim, num_classes)
        opt = torch.optim.Adam(log.parameters(), lr=0.1)
        log.to(embeds.device)
        np.random.seed(epoch)
        torch.manual_seed(epoch)
        torch.cuda.manual_seed(epoch)
        random.seed(epoch)
        
        
        
        log.train()
        opt.zero_grad()
            
            
        logits = log(embeds, split_edges['train']['edge'], training_negative)
        loss = xent(logits, split_edges['train']['label'])

        loss.backward(retain_graph=True)
        opt.step()

            
        auc, hits, ap = evaluate_model(log, embeds, split_edges['test']['edge'], split_edges['test']['edge_neg'],
                                                      None,
                                                      split_edges['test']['label'], test=True)
           
        if auc > best_auc:
            best_auc = auc

            
            print('Epoch:', epoch)
            print("Best auc:", best_auc)
            print("Best ap:", float(ap))
        auc_list.append(auc)
        ap_list.append(float(ap))
        

    return auc_list, ap_list, hits_list

def run_similarity_search(test_embs, test_lbls):
    numRows = test_embs.shape[0]

    cos_sim_array = pairwise.cosine_similarity(test_embs) - np.eye(numRows)
    st = []
    for N in [5, 10, 20, 50, 100]:
        indices = np.argsort(cos_sim_array, axis=1)[:, -N:]
        tmp = np.tile(test_lbls, (numRows, 1))
        selected_label = tmp[np.repeat(np.arange(numRows), N), indices.ravel()].reshape(numRows, N)
        original_label = np.repeat(test_lbls, N).reshape(numRows,N)
        st.append(str(np.round(np.mean(np.sum((selected_label == original_label), 1) / N), 4)))

    st = ','.join(st)
    print("\t[Similarity] [5,10,20,50,100] : [{}]".format(st))


def run_kmeans(x, y, k):
    estimator = KMeans(n_clusters=k)

    NMI_list = []
    for i in range(10):
        estimator.fit(x)
        y_pred = estimator.predict(x)
        s = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
        NMI_list.append(s)

    mean = np.mean(NMI_list)
    std = np.std(NMI_list)
    print('\t[Clustering] NMI: {:.4f} | {:.4f}'.format(mean, std))
    return NMI_list
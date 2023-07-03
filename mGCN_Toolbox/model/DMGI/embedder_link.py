# the code is based on https://github.com/pcy1302/DMGI/blob/master/evaluate.py
import torch
import torch.nn as nn

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, pairwise, f1_score
from sklearn.metrics import roc_auc_score
from mGCN_Toolbox.model.DMGI.logreg_link import LogReg
import random

def area_under_prc(pred, target):
    """
    Area under precision-recall curve (PRC).
    Parameters:
        pred (Tensor): predictions of shape :math:`(n,)`
        target (Tensor): binary targets of shape :math:`(n,)`
    """
    order = pred.argsort(descending=True)
    target = target[order]
    print(type(target))
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

def evaluate_model(model_GNN, embeds, edge, edge_neg, common_neighbors, labels, test=False, num_pos=None):
    model_GNN.eval()
    logits = model_GNN(embeds, edge, edge_neg)
    return evaluate_metrics(labels.cpu().numpy(), torch.sigmoid(logits).cpu().detach().numpy(), test, edge.shape[0])



def evaluate(embeds, split_edges, isTest=True):
    training_negative = split_edges['train']['edge_neg'][np.random.randint(0, split_edges['train']['edge_neg'].shape[0], split_edges['train']['edge'].shape[0])]
    # train_embs = embeds[idx_train]
    xent = nn.BCEWithLogitsLoss()
    # val_embs = embeds[idx_val]
    # test_embs = embeds[idx_test]
    #
    # train_lbls = torch.argmax(labels[idx_train], dim=1)
    # val_lbls = torch.argmax(labels[idx_val], dim=1)
    # test_lbls = torch.argmax(labels[idx_test], dim=1)

    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []


    for epoch in range(3):
        #print(embeds.shape)
        log = LogReg(embeds.shape[-1], 2)
        opt = torch.optim.Adam(log.parameters(), lr=0.1)
        log.to(embeds.device)
        np.random.seed(epoch)
        torch.manual_seed(epoch)
        torch.cuda.manual_seed(epoch)
        random.seed(epoch)
        auc_list = []
        ap_list = []
        hits_list = []
        best_val = 0
        best_test = 0
        best_hits = [0, 0, 0, 0, 0, 0, 0]
        t = embeds
        embeds = np.reshape(t,(3025,64))
        #print("embeds:",embeds.shape)
        for iter_ in range(1000):
            # train
            log.train()
            opt.zero_grad()
            #print(embeds)
            #print(split_edges['train']['edge'].shape)
            #print(training_negative.shape)
            logits = log(embeds, split_edges['train']['edge'], training_negative)
            loss = xent(logits, split_edges['train']['label'])

            loss.backward()
            opt.step()

            # val
            #print(type(split_edges['valid']['label']))
            #print(split_edges['valid']['label'].shape)
            eval = evaluate_model(log, embeds, split_edges['valid']['edge'], split_edges['valid']['edge_neg'], None,
                                  split_edges['valid']['label'])
            # if epoch == epochs - 1:
            #     temp = evaluate_model(split_edges['test']['edge'], split_edges['test']['edge_neg'], None,
            #                           split_edges['test']['label'], test=True)
            #     print("AUC last:", temp)

            if eval > best_val:
                best_val = eval

                best_test, best_hits, best_ap = evaluate_model(log, embeds, split_edges['test']['edge'], split_edges['test']['edge_neg'],
                                                      None,
                                                      split_edges['test']['label'], test=True)
                print('Epoch:', epoch)
                print("Best Validation:", best_val)
                print("Best Test:", best_test)

        print(best_hits)
        print("Best ap:", best_ap)
        auc_list.append(best_test)
        ap_list.append(best_ap)
        hits_list.append(best_hits)

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


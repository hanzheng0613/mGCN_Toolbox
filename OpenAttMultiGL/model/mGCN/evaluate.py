# the code is based on https://github.com/pcy1302/DMGI/blob/master/evaluate.py

"""
   PyTorch implementation of HDMI: High-order Deep Multiplex Infomax  
   
       https://github.com/baoyujing/HDMI/tree/master
        
"""

import torch
import torch.nn as nn

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, pairwise, f1_score


from OpenAttMultiGL.model.hdmi.logreg import LogReg

def evaluate(embeds, idx_train, idx_val, idx_test, labels, isTest=True):
    nb_classes = labels.shape[1]
    train_embs = embeds[idx_train]
    xent = nn.CrossEntropyLoss()
    val_embs = embeds[idx_val]
    test_embs = embeds[idx_test]

    train_lbls = torch.argmax(labels[idx_train], dim=1)
    val_lbls = torch.argmax(labels[idx_val], dim=1)
    test_lbls = torch.argmax(labels[idx_test], dim=1)


    for _ in range(50):
        log = LogReg(train_embs.shape[1], nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=0.1)
        log.to(train_lbls.device)


        for iter_ in range(100):
            # train
            log.train()
            opt.zero_grad()
            logits = log(train_embs)
            loss = xent(logits, train_lbls)
            
            
            loss.backward()
            opt.step()
            #logits = log(train_embs)


    test_embs = np.array(test_embs.cpu())
    test_lbls = np.array(test_lbls.cpu())
    run_kmeans(test_embs, test_lbls,nb_classes)
    run_similarity_search(test_embs, test_lbls)
    
def run_similarity_search(test_embs, test_lbls):
    numRows = test_embs.shape[0]
    sim = []
    cos_sim_array = pairwise.cosine_similarity(test_embs) - np.eye(numRows)
    st = []
    for N in [5, 10, 20, 50, 100]:
        indices = np.argsort(cos_sim_array, axis=1)[:, -N:]
        tmp = np.tile(test_lbls, (numRows, 1))
        selected_label = tmp[np.repeat(np.arange(numRows), N), indices.ravel()].reshape(numRows, N)
        original_label = np.repeat(test_lbls, N).reshape(numRows,N)
        st.append(str(np.round(np.mean(np.sum((selected_label == original_label), 1) / N), 4)))
    for i in st:
        sim.append(float(i))
    st = ','.join(st)
    
    sim_mean = np.mean(sim)
    print("\t[Similarity] [5,10,20,50,100] : [{}]".format(st))
    #return sim

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
    print('\t[Clustering] NMI: {:.4f} | {:.4f}'.format(mean, std))
    #return NMI_list
'''
     The PyTorch implementation of X-GOAL: Multiplex Heterogeneous Graph Prototypical Contrastive Learning\
     
     https://github.com/baoyujing/X-GOAL/tree/main    
'''

import torch
import torch.nn as nn

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, pairwise, f1_score

from OpenAttMultiGL.model.X_GOAL import LogReg


def evaluate(embeds,idx_train, idx_val, idx_test,labels,nb_classes,isTest=True):
    print("Evaluating...")
    
    #nb_classes = labels.shape[1]
    train_embs = embeds[idx_train.long()]
    #print('type1',type(idx_train))
    xent = nn.CrossEntropyLoss()
    #print('type2',type(idx_val))
    val_embs = embeds[idx_val.long()]
    #idx_test=torch.LongTensor(idx_test)
    #print('type3',type(idx_test))
    test_embs = embeds[idx_test.long()]

    train_lbls = torch.argmax(labels[idx_train], dim=1)
    val_lbls = torch.argmax(labels[idx_val], dim=1)
    test_lbls = torch.argmax(labels[idx_test], dim=1)


    xent = nn.CrossEntropyLoss()
   
    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []
    k1_list = []
    sim_list = []
    
    
    
    for _ in range(50):
        log = LogReg(train_embs.shape[1], nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=0.1)
        log.to(train_lbls.device)

        val_accs = []; test_accs = []
        val_micro_f1s = []; test_micro_f1s = []
        val_macro_f1s = []; test_macro_f1s = []

        for iter_ in range(100):
            # train
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

            # val
            logits = log(val_embs)
            preds = torch.argmax(logits, dim=1)

            val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
            val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
            val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')

            val_accs.append(val_acc.item())
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)

            # test
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)

            test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
            test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')

            test_accs.append(test_acc.item())
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)
            
            #test_embs = np.array(test_embs.cpu())
            #test_lbls = np.array(test_lbls.cpu())
            
    
        
        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])

        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])
        macro_f1s_val.append(val_macro_f1s[max_iter])

        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])

    if isTest:
        print("\t[Classification] Macro-F1: {:.4f} ({:.4f}) | Micro-F1: {:.4f} ({:.4f})".format(
            np.mean(macro_f1s), np.std(macro_f1s), np.mean(micro_f1s), np.std(micro_f1s)))
    else:
        return np.mean(macro_f1s_val), np.mean(macro_f1s)

    test_embs = np.array(test_embs.cpu())
    test_lbls = np.array(test_lbls.cpu())
    
    #test_embs = np.array(test_embs) for jupyter notebook
    #test_lbls = np.array(test_lbls)
    k1 = run_kmeans(test_embs, test_lbls, nb_classes)
    sim = run_similarity_search(test_embs, test_lbls)
    #sim = sim[0]
    #sim_list.append(sim)
    #k1_list.append(k1)
    

    
    return macro_f1s, micro_f1s, k1,sim


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
    return st


def run_kmeans(x, y, k):
    estimator = KMeans(n_clusters=k,n_init=10)#, n_jobs=16)

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
